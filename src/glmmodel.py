import warnings 
warnings.filterwarnings("ignore")
import paddle
from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMTokenizer,
)

def get_model():
    model_name_or_path = 'THUDM/chatglm-6b'
    tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path)
    config = ChatGLMConfig.from_pretrained(model_name_or_path)
    paddle.set_default_dtype(config.paddle_dtype)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_name_or_path,
        tensor_parallel_degree=paddle.distributed.get_world_size(),
        tensor_parallel_rank=0,
        load_state_as_np=True,
        dtype=config.paddle_dtype,
    )
    model.eval()
    return tokenizer, model
