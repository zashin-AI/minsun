#tensorboard는 그래프를 넷상에서 볼 수 있게 한 것. 시각화라 성능과는 상관없음.

# # cmd -> 
# cd \
# cd nmb
# cd nmb_data
# cd graph
# tensorboard --logdir=.
# (텐서보드 빼기 로그dir= .은 현재폴더)
# 위의 순서대로 입력하고 enter누르기

# 인터넷 켜서 주소창에 
# 127.0.0.1:6006 ( 127.0.0.1=> 로컬 주소.내 컴퓨터 주소라는 뜻 
#                     : 6006=> 로컬호스트에 텐서보드 번호)



# log_dir : TensorBoard에서 로그 파일을 저장할 디렉토리의 경로
# histogram_freq :모델의 계층에 대한 활성화 및 가중치 히스토그램을 계산할 빈도 (에포크 단위).                           
#                 0으로 설정하면 히스토그램이 계산되지 않는다.
#                 히스토그램 시각화를 위해 유효성 검사 데이터 (또는 분할)를 지정해야한다.                          
#                 *histogram= 통계 등 자료의 빈도 분포 특성을 시각화하는 도구  
# write_graph : TensorBoard 에서 그래프를 시각화할지 여부. write_graph가 True로 설정되면 로그 파일이 상당히 커질 수 있다.
# write_images : TensorBoard 에서 이미지로 시각화하기 위해 모델 가중치를 쓸지 여부.



'''
#시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설치
plt.figure(figsize=(10,6))  # 판 깔아주는 것.
plt.subplot(2,1,1) #(2행 1열 중 첫번째)
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
# subplot은 두 개의 그림을 그린다는 것. plot은 도화지 하나라고 생각.
plt.title('한글') # plt.title('손실비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2,1,2) #(2행 2열 중 2번째)
plt.plot(hist.history['acc'],marker='.', c='red', label='acc') #metrics의 이름과 똑같이 넣기
# 그림보면 갱신되는 점은 그대로 두고 뒤에 값 올라간 점은 없어도 된다. 
plt.plot(hist.history['val_acc'],marker='.', c='blue', label='val_acc')
plt.grid() # 격자. 모눈종이 형태. 바탕을 그리드로 하겠다는 것. 

plt.title('accuracy') # plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

'''