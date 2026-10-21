from gguf_parser import GGUFParser

parser = GGUFParser("E:/models/qwen2-1_5b-instruct-fp16.gguf")
parser.parse()

# 打印所有 tensor 名称和 shape
parser.print()
