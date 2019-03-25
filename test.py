from PIL import Image
import tensorflow as tf
from cassandra.cluster import Cluster
import time
from flask import Flask,request,redirect, flash
from werkzeug.utils import secure_filename
import os

import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)


UPLOAD_FOLDER ='/app'   #'/home/hchen/桌面/project'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
localtime = time.strftime("%Y-%m-%d %H:%M:%S")
KEYSPACE = "imagesapce"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))              
        result = useModel(f.filename)
        insertValues(localtime,f.filename,result)
        return "The number in the picture is {}".format(result)
        
    
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
    </head>
    <body>
        <h1>Please upload your picture.</h1>
        <form action="" enctype='multipart/form-data' method='POST'>
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''
    
def imageprepare(file_name):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    
    im = Image.open(file_name)  
    im = im.convert('L')    
    tv = list(im.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    print(tva)
    return tva

    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """






def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def GetData():
    x = tf.placeholder(tf.float32, [None, 784])

    y_ = tf.placeholder(tf.float32, [None, 10])
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x,[-1,28,28,1])
    

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv, keep_prob, h_conv2, x, y_

def useModel(file_name):
    result = imageprepare(file_name)
    y_conv, keep_prob,h_conv2, x, y_ = GetData()
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/app/model.ckpt")  #"/home/haochen/桌面/BigData/Docker/model.ckpt")#这里使用了之前保存的模型参数
        #print ("Model restored.")
    
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
        print(h_conv2)
        print('result:')
        print(predint[0])
        return predint[0]

def insertValues(localtime, filename, result):
    cluster = Cluster(contact_points=['0.0.0.0'],port=9042)
    session = cluster.connect()
    try:

       session.execute("""

           CREATE KEYSPACE %s

           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }

           """ % KEYSPACE)


       log.info("setting keyspace...")

       session.set_keyspace(KEYSPACE)


       log.info("creating table...")

       session.execute("""

           CREATE TABLE mytable (

               time text,

               filename text,

               result int,

               PRIMARY KEY (time, filename, result)

           )

           """) 
       session.execute("""
                          
                INSERT INTO mytable (time,filename,result) 
    
                VALUES (%s, %s, %s)
    
        	   """ ,(localtime,filename,result))

    except Exception as e:

       log.error("Unable to create keyspace")

       log.error(e)
        
        
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
