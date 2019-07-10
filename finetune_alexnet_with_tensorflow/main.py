from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from test import test_image
import urllib.request
from bs4 import BeautifulSoup

from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/test_img', secure_filename(f.filename))
        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/test_img', 'test.jpg'), img)
        class_name=test_image(upload_path, num_class=102)
        # class_name = 'water lily'
        fo = open("url.txt", "r")
        for line in fo.readlines():
            temp = line.split(',')
            if temp[0] == class_name:
                url = temp[1]
        a = urllib.request.urlopen(url)
        html = a.read()  # 读取网页源码
        soup = BeautifulSoup(html, 'lxml')  # 生成标签树

        name = soup.find('h1')
        name = name.get_text()

        text = soup.find('div', class_="para")
        text = text.get_text()

        return render_template('recognition_ok.html', userinput=user_input, val1=time.time(),
                               classname=class_name, name=name, text=text, url=url)

    return render_template('recognition.html')


if __name__ == '__main__':
    app.run(debug=True)
