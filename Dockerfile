FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \ 
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user . .

EXPOSE 7860

CMD ["bash","./start.sh"]
