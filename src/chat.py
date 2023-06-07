import paddle
import random

class Chater(object):
    context = "你是一个产品经理，下面的文字是用户提出的需求。请把需求整理成为一个PRD文档，需要清晰易懂，有条理，便于开发人员理解，可以直接用于系统开发：\n用户需求\n\n```{content}```"
    input_length = 2048
    output_length = 2048
    temperature = random.random()

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __prompt_compose(self, content):
        return self.context.format(content=content)
    
    def generate(self, content):
        prompt = self.__prompt_compose(content)
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            max_length=self.input_length,
            truncation=True,
            truncation_side="left",
        )
        input_map = {}
        for key in inputs:
            input_map[key] = paddle.to_tensor(inputs[key])
        infer_result = self.model.generate(
            **input_map,
            decode_strategy="sampling",
            top_k=20 ,
            max_length=self.output_length,
            # use_cache=True,
            use_fast=True,
            use_fp16_decoding=True,
            repetition_penalty=1.2,
            temperature = self.temperature,
            length_penalty=1,
        )[0]
        res = self.tokenizer.decode(infer_result.tolist()[0], skip_special_tokens=True)
        res = res.strip("\n")
        return res