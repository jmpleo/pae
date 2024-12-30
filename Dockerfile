#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
#FROM pytorch/manylinux-cpu:latest
FROM cnstark/pytorch:1.10.1-py3.9.12-ubuntu20.04

COPY . .

RUN python3 -m pip install numpy huggingface-hub

ENTRYPOINT ["python3", "sample.py"]

CMD ["--help"]

