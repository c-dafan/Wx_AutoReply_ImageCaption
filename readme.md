# 微信自动回复Py服务器端

## abstract

这个项目需要配合autoReply_WX使用。

主体功能为：微信公众号自动回复。微信公众号将消息转发到autoReply_WX后，由autoReply_wx将各类消息路由
到各自个handler。autoReply_Wx使用SpringBoot框架。

这个项目为对图像内容和视频内容的识别，并对内容进行描述（image caption）。
由autoReply_WX将**视频或者图像消息**发送到RabbitMq，此项目作为接收端，处理接收到的消息，
并把处理结果返回给autoReply_Wx,由其发送给用户。

autoReply_Wx利用RabbitMq，从而实现把消息路由到此接收端。

## 使用方法


需要将setting.py 中的以下配置修改为自己的RabbitMq的host:port
```
username = "guest"
password = "guest"
host = "106.13.50.135"
port = "5672"
```

然后

```bash
python main.py
```
