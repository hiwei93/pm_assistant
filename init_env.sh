git clone https://gitee.com/paddlepaddle/PaddleNLP.git -b develop

cd PaddleNLP
pip install -e . --user

# 下次重启该项目后，可能要再安装paddlepaddle-gpu==0.0.0.post112，或者加 --user 避免下次再安装
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html --user