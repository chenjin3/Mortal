import torch
import tensorrt as trt
import numpy as np
from model import Brain, DQN
from config import config as config_dict

def get_sample_input() -> torch.Tensor:
    """生成模型所需的样本输入"""
    batch_size = 1
    in_channels = 1012  
    seq_length = 34     
    
    dummy_input = torch.zeros((batch_size, in_channels, seq_length), dtype=torch.float32)
    return dummy_input

def convert_to_trt():
    # 加载权重
    state = torch.load(config_dict['control']['state_file'], weights_only=True)
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks'] 
    conv_channels = cfg['resnet']['conv_channels']

    # 创建模型
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    # 获取正确格式的样本输入
    dummy_input = get_sample_input()
    
    # 导出ONNX
    torch.onnx.export(
        mortal, 
        dummy_input,
        "mortal.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}},
        verbose=True,
        opset_version=11
    )
    
    # 创建TRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建builder和network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建解析器
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX
    with open("mortal.onnx", 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse mortal.onnx")
    
    # 构建引擎配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 获取输入张量的名称和形状
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    print(f"Input tensor name: {input_tensor.name}")
    print(f"Input shape: {input_shape}")

    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    
    # 设置输入张量的动态范围
    min_shape = (1, 1012, 34)
    opt_shape = (1, 1012, 34)
    max_shape = (1, 1012, 34)
    
    profile.set_shape(
        input_tensor.name,  # 使用实际的输入张量名称
        min_shape,         # 最小尺寸
        opt_shape,         # 优化尺寸
        max_shape         # 最大尺寸
    )
    
    # 添加优化配置文件
    config.add_optimization_profile(profile)

    # 构建引擎
    try:
        print("Building serialized network...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized network")
        
        print("Saving engine...")
        with open("model.trt", "wb") as f:
            f.write(serialized_engine)
            
    except Exception as e:
        print(f"Error during engine building: {str(e)}")
        raise
    
    print("TensorRT model conversion completed successfully!")

def test_model_forward():
    """测试模型前向传播"""
    state = torch.load(config_dict['control']['state_file'], weights_only=True)
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks'] 
    conv_channels = cfg['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    mortal.load_state_dict(state['mortal'])
    
    dummy_input = get_sample_input()
    print("Model input shape:", dummy_input.shape)
    
    # 打印每一层的输出形状
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: input shape = {input[0].shape}, output shape = {output.shape}")
    
    # 注册钩子
    hooks = []
    for name, module in mortal.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear, torch.nn.Flatten)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # 尝试前向传播
    with torch.no_grad():
        output = mortal(dummy_input)
        print("Model output shape:", output.shape)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    try:
        # 获取TensorRT版本
        print(f"TensorRT version: {trt.__version__}")
        
        # 首先测试模型
        test_model_forward()
        # 如果测试成功，进行转换
        convert_to_trt()
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
