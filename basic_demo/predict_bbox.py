import os
import json
import torch
import argparse
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
import re
TOKENIZER_PATH = "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/modle_weight/vicuna-7b-v1.5"

def custom_collate_fn(batch):
    """
    自定义的collate函数，用于处理PIL.Image对象
    """
    id_list = [item['id'] for item in batch]
    image_list = [item['image'] for item in batch]
    conversations_list = [item['conversations'] for item in batch]
    original_image_path_list = [item['original_image_path'] for item in batch]
    
    return {
        'id': id_list,
        'image': image_list,
        'conversations': conversations_list,
        'original_image_path': original_image_path_list

    }
class BboxDataset(Dataset):
    """
    用于加载预处理好的图像和标注数据的数据集
    """
    def __init__(self, json_file, image_base_dir):
        """
        Args:
            json_file (str): 包含图像路径和标注的JSON文件路径
            image_base_dir (str): 图像文件的基础目录
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_base_dir = image_base_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_base_dir, item['image'])
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个占位图像
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 深度复制会话数据，防止在处理过程中修改原始数据
        conversations = []
        for conv in item['conversations']:
            conversation = {}
            for key, value in conv.items():
                conversation[key] = value.copy() if isinstance(value, list) else value
            conversations.append(conversation)
            
        return {
            'id': item['id'],
            'image': image,
            'conversations': conversations,
            'original_image_path': item['image']
        }


def parse_bbox_from_response(response):
    """
    从模型响应中解析边界框坐标
    """
    # 寻找[x, y, x2, y2]格式的坐标，包括整数和小数
    pattern = r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
    match = re.search(pattern, response)
    
    if match:
        try:
            x1 = int(match.group(1))  # 提取整数值
            y1 = int(match.group(2))
            x2 = int(match.group(3))
            y2 = int(match.group(4))
            # 验证坐标有效性，如果仍需要确保有效性范围
            if x1 <= x2 and y1 <= y2:
                x1 = x1/1000
                x2 = x2/1000
                y1 = y1/1000
                y2 = y2/1000
                return [x1, y1, x2, y2]
        except ValueError:
            pass
    
    # 不符合条件的返回None
    return None


def generate_bbox_prompt(element):
    """
    为列表元素生成提示，用于请求模型生成边界框
    """
    return (f"Can you point out '{element}' in the image and provide the bounding boxes of their location?")


def setup_distributed(rank, world_size):
    """
    设置分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置GPU设备
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()


def run_inference(rank, world_size, args):
    """
    在每个设备上运行推理
    """
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 加载模型和分词器
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(rank)
    
    # DDP模型包装
    model = DDP(model, device_ids=[rank])
    
    # 准备数据集和DataLoader
    dataset = BboxDataset(args.data_json, args.image_dir)
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # 设置为1，每次处理一个样本
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    # 存储处理后的结果
    processed_samples = []
    
    # 进行推理
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"GPU {rank} processing"):
            sample_id = batch['id'][0]
            image = batch['image'][0]
            conversations = batch['conversations'][0]  # 一批中的第一个样本
            image_path = batch['original_image_path'][0]

            image_path = os.path.join(args.image_dir, image_path)
            image = Image.open(image_path).convert('RGB')
            processed_conversations = []
            
            for conversation in conversations:
                processed_conv = {}
                
                # 处理 <CAPTION> 列表
                if "<CAPTION>" in conversation and conversation["<CAPTION>"]:
                    caption_elements = conversation["<CAPTION>"]
                    caption_with_bbox = []
                    
                    for element in caption_elements:
                        prompt = generate_bbox_prompt(element)
                        
                        # 使用CogVLM生成边界框
                        # response = model.module.chat(
                        #     tokenizer,
                        #     query=prompt,
                        #     image=image,
                        #     history=None,
                        #     max_new_tokens=args.max_tokens,
                        #     do_sample=False
                        # )
                        input_by_model = model.module.build_conversation_input_ids(
                            tokenizer, 
                            query=prompt,
                            history=None, 
                            images=[image]
                        )
                        
                        inputs = {
                            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(rank),
                            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(rank),
                            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(rank),
                            'images': [[input_by_model['images'][0].to(rank).to(torch.bfloat16)]] if image is not None else None,
                        }
                        
                        if 'cross_images' in input_by_model and input_by_model['cross_images']:
                            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(rank).to(torch.bfloat16)]]
                        
                        # 生成参数
                        gen_kwargs = {
                            "max_length": args.max_tokens,
                            "do_sample": False
                        }
                        
                        # 生成响应
                        outputs = model.module.generate(**inputs, **gen_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        response = tokenizer.decode(outputs[0])
                        # print('image_path:', image_path, 'prompt:', prompt)
                        # print('response:', response)
                        response = response.split("</s>")[0]

                        # 解析边界框
                        bbox = parse_bbox_from_response(response)
                        
                        caption_with_bbox.append({
                            "content": element,
                            "bbox": bbox
                        })
                        # print(response)
                        print(bbox)
                    processed_conv["<CAPTION>"] = caption_with_bbox
                else:
                    processed_conv["<CAPTION>"] = None
                
                # 处理 <REASONING> 列表
                if "<REASONING>" in conversation and conversation["<REASONING>"]:
                    reasoning_elements = conversation["<REASONING>"]
                    reasoning_with_bbox = []
                    
                    for element in reasoning_elements:
                        prompt = generate_bbox_prompt(element)
                        
                        # 使用CogVLM生成边界框
                        # response = model.module.chat(
                        #     tokenizer,
                        #     query=prompt,
                        #     image=image,
                        #     history=None,
                        #     max_new_tokens=args.max_tokens,
                        #     do_sample=False
                        # )
                        input_by_model = model.module.build_conversation_input_ids(
                            tokenizer, 
                            query=prompt,
                            history=None, 
                            images=[image]
                        )
                        
                        inputs = {
                            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(rank),
                            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(rank),
                            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(rank),
                            'images': [[input_by_model['images'][0].to(rank).to(torch.bfloat16)]] if image is not None else None,
                        }
                        
                        if 'cross_images' in input_by_model and input_by_model['cross_images']:
                            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(rank).to(torch.bfloat16)]]
                        
                        # 生成参数
                        gen_kwargs = {
                            "max_length": args.max_tokens,
                            "do_sample": False
                        }
                        
                        # 生成响应
                        outputs = model.module.generate(**inputs, **gen_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        response = tokenizer.decode(outputs[0])
                        response = response.split("</s>")[0]
                        # 解析边界框
                        bbox = parse_bbox_from_response(response)
                        print(bbox)
                        reasoning_with_bbox.append({
                            "content": element,
                            "bbox": bbox
                        })
                    
                    processed_conv["<REASONING>"] = reasoning_with_bbox
                else:
                    processed_conv["<REASONING>"] = None
                
                # 添加其他可能存在的键
                for key, value in conversation.items():
                    if key not in ["<CAPTION>", "<REASONING>"]:
                        processed_conv[key] = value
                
                processed_conversations.append(processed_conv)
            
            # 保存处理后的样本
            processed_samples.append({
                "id": sample_id,
                "image": image_path,
                "conversations": processed_conversations
            })
            
            # 定期保存中间结果(每个GPU独立保存)
            if args.save_intermediate and len(processed_samples) % args.save_every == 0:
                intermediate_path = f"{args.output_file}.rank{rank}.part{len(processed_samples)//args.save_every}"
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_samples, f, ensure_ascii=False, indent=4)
                print(f"Rank {rank}: Intermediate results saved to {intermediate_path}")
    
    # 收集所有GPU上的结果
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, processed_samples)
    
    # 在rank 0上保存结果
    # 修改代码中的合并结果部分
    if rank == 0:
        # 读取原始输入文件以获取正确的顺序
        with open(args.data_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 创建ID到结果的映射
        result_map = {item["id"]: item for sublist in all_results for item in sublist}
        
        # 按原始顺序重建结果
        merged_results = []
        for item in original_data:
            if item["id"] in result_map:
                merged_results.append(result_map[item["id"]])
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to {args.output_file}")
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVLM Distributed Inference for BBOX Generation")
    parser.add_argument("--model_path", type=str, default="/mnt/pfs-mc0p4k/nlu/team/yuhaofu/modle_weight/cogvlm", help="Path to CogVLM model")
    parser.add_argument("--data_json", type=str, required=True, help="Path to preprocessed data JSON file")
    parser.add_argument("--image_dir", type=str, default="/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k", help="Base directory of images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results with bbox annotations")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers per GPU")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for generation")
    parser.add_argument("--save_intermediate", action="store_true", help="Whether to save intermediate results")
    parser.add_argument("--save_every", type=int, default=1, help="Save intermediate results every N samples")
    
    args = parser.parse_args()
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        if args.world_size > torch.cuda.device_count():
            print(f"Warning: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available")
            args.world_size = torch.cuda.device_count()
        
        # 启动多进程
        import torch.multiprocessing as mp
        mp.spawn(run_inference, args=(args.world_size, args), nprocs=args.world_size)
    else:
        print("No GPU available. This script requires at least one GPU.")