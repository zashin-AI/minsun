https://tech.10000lab.xyz/naturallanguageprocessing/how-to-run-gpt-2.html
$ git clone https://github.com/openai/gpt-2.git
$ cd gpt-2

https://tech.10000lab.xyz/naturallanguageprocessing/how-to-run-gpt-2.html


$ pip install tensorflow==1.15
$ pip install -r requirements.txt

$ python download_model.py 124M
$ python download_model.py 355M
$ python download_model.py 774M
$ python download_model.py 1558M

python src/interactive_conditional_samples.py --model_name="355M"