import tensorflow as tf
import matlab.engine
import numpy as np
import pyaudio
import matlab
import pygame
import time
import math
import wave
import os

#地址
wav_files='wav\\'
fbank_files='feature\\'
txt_files='feature-translated\\'
#录制十次，每次五秒
def record():
	for x in range(0,4):
		CHUNK = 1024
		FORMAT = pyaudio.paInt16
		CHANNELS = 1
		RATE = 16000
		RECORD_SECONDS = 3
		WAVE_OUTPUT_FILENAME =wav_files+'test_'+'%04d' % x +'.WAV'
		p = pyaudio.PyAudio()
		stream = p.open(format=FORMAT,
		                channels=CHANNELS,
		                rate=RATE,
		                input=True,
		                frames_per_buffer=CHUNK)
		print("* recording")
		frames = []
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		    data = stream.read(CHUNK)
		    frames.append(data)
		print("* done recording")
		stream.stop_stream()
		stream.close()
		p.terminate()
		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()
	#fbank特征提取
	os.system('Hcopy -A -D -C  config.cfg -S wav.scp')
def zhuanhua():
    for root1,dirs1,files1 in os.walk(fbank_files):
        for OneFileName1 in files1:
            eng = matlab.engine.start_matlab('MATLAB_R2016b')
            ans = eng.readhtk(fbank_files+OneFileName1)
            L = []
            b = ''
            for i in range(len(ans)):
                b = ''
                for j in range(40):
                    if j==39:
                        b=b+str(ans[i][j])
                    else:
                        b = b + str(ans[i][j])+" "
                L.append(b)
            L1=[]
            #print(L)
            #input()
            for x in range(len(L)):
                if x-30<0:
                    k=0
                else:
                    k=x-30
                first=L[k]
                for fro in range(1,30):
                    if x-30+fro<0:
                        k=0
                    else:
                        k=x-30+fro
                    first=first+' '+L[k]
                if x+1>=len(L):
                    t=len(L)-1
                else:
                    t=x+1
                back=L[t]
                for bac in range(2,11):
                    if x+bac>=len(L):
                        t=len(L)-1
                    else:
                        t=x+bac
                    back=back+' '+L[t]
                zong=first+' '+L[k]+' '+back+'\n'
                L1.append(zong)
            #print(L1)
            #input()
            writer=open(txt_files+OneFileName1+'.txt','w')
            for line in range(len(L1)):
                writer.write(L1[line])
    		#input()
            writer.close()
#添加层
def add_layer(inputs,in_size,out_size,activation_function):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
         outputs=activation_function(Wx_plus_b)
    return outputs
def zhixin(x_data,y_prediction):
	confidence=[]
	p1=[]
	p2=[]
	for j in range(1,len(x_data)+1):
		if j-29>1:
			hs=j-29
		else:
			hs=1
		sum1=0
		sum2=0
		for k in range(hs,j+1):
			sum1=sum1+y_prediction[k-1][1]
			sum2=sum2+y_prediction[k-1][2]
		p1k=sum1/(j-hs+1)
		p2k=sum2/(j-hs+1)
		p1.append(p1k)
		p2.append(p2k)
	for j in range(1,len(x_data)+1):
		if j-99>1:
			hm=j-99
		else:
			hm=1
		p1max=0
		p2max=0
		for k in range(hm,j+1):
			if p1[k-1]>=p1max:
				p1max=p1[k-1]
			if p2[k-1]>=p2max:
				p2max=p2[k-1]
		con=math.sqrt(p1max*p2max)
		confidence.append(con)
	return confidence
def panduan(confidence,juezhi,tiaojian):
	a=0
	for i in range(len(confidence)):
		if confidence[i]>=juezhi:
			a=a+1
		else:
			a=0
		if a==tiaojian:
			break
	if a==tiaojian:
		return 1
	else:
		return 0
#播放唤醒成功
def wake_up_success():
	num=str(np.random.randint(1,4))
	file=r'yinpin/'+num+'.mp3'
	pygame.mixer.init()
	print("播放")
	track = pygame.mixer.music.load(file)
	pygame.mixer.music.play()
	time.sleep(1)
	pygame.mixer.music.stop()

#构建网络
x=tf.placeholder(tf.float32,[None,1640])
y=tf.placeholder(tf.float32,[None,3])
L1=add_layer(x,1640,128,activation_function=tf.nn.tanh)
L2=add_layer(L1,128,32,activation_function=tf.nn.relu)
prediction=add_layer(L2,32,3,activation_function=None)
zuizhong=tf.nn.softmax(prediction)
saver=tf.train.Saver()
#主程序从此开始
record()
zhuanhua()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,"2")
	huan=[]
	for root,dirs,files in os.walk(txt_files):
		for OneFileName in files:
			f=np.loadtxt(txt_files+OneFileName)
			x_data=f
			y_prediction=sess.run(zuizhong,feed_dict={x:x_data})
			confidence=[]
			confidence=zhixin(x_data,y_prediction)
			huanxing=panduan(confidence,0.331,10)
			huan.append(huanxing)
	for i in range(len(huan)):
		if(huan[i]==1):
			wake_up_success()
		else:
			print('sorry?')
