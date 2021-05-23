from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import pad, maximum, random, int32

#Original WaveGAN: https://github.com/chrisdonahue/wavegan
#Label embeding using the method in https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/


# 생성자
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=4, padding='same'
                    , name = '1DTConv', activation = 'relu'):
    x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                        name = name, activation = activation)(K.expand_dims(input_tensor, axis=1)) 
                        # input_tensor의 쉐이프 차원에서 두번째 차원을 추가하여 확장한다.
    x = K.squeeze(x, axis=1) 
    # x의 쉐이프의 차원 중 사이즈 1인 것을 찾아서 제거한다.?
    return x

# 임의의 난수 100개 설정
def generator(z_dim = 100,
              architecture_size = 'audio_size',
              n_classes = 2):
        
    generator_filters = [1024, 512, 256, 128, 64]

    label_input = Input(shape=(1,), dtype='int32', name='generator_label_input')
    label_em = Embedding(n_classes, n_classes * 20, name = 'label_embedding')(label_input) 
    # 양의 정수 (인덱스)를 고정 된 크기의 조밀한 벡터로 변환합니다. 임베딩 레이어로 벡터화(희소 표현된 단어를 임의의 길이의 실수 벡터로 표현)
    label_em = Dense(16, name = 'label_dense')(label_em)
    label_em = Reshape((16, 1), name = 'label_respahe')(label_em)
    
    generator_input = Input(shape=(z_dim,), name='generator_input')
    x = generator_input

    if architecture_size == 'audio_size':
        x = Dense(32768, name='generator_input_dense')(x)
        x = Reshape((16, 2048), name='generator_input_reshape')(x)
        
    x = ReLU()(x)
    
    x = Concatenate()([x, label_em]) 
    
    if architecture_size == 'audio_size':
        #layer 0 to 4
        for i in range(5):
            x = Conv1DTranspose(
                input_tensor = x
                , filters = generator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding='same'
                , name = f'generator_Tconv_{i}'
                , activation = 'relu'
                )
        
        #layer 5
        x = Conv1DTranspose(
            input_tensor = x
            , filters = 1
            , kernel_size = 25
            , strides = 7
            , padding='same'
            , name = 'generator_Tconv_5'
            , activation = 'tanh'
            ) 
            # strides 7로 설정하여 110,250 오디오 길이를 커버할 수 있도록 바꿔준다.
    generator_output = x 
    generator = Model([generator_input, label_input], generator_output, name = 'Generator')
    return generator

model = generator()
model.summary()
'''
Model: "Generator"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
generator_input (InputLayer)    [(None, 100)]        0
__________________________________________________________________________________________________
generator_label_input (InputLay [(None, 1)]          0
__________________________________________________________________________________________________
generator_input_dense (Dense)   (None, 32768)        3309568     generator_input[0][0]
__________________________________________________________________________________________________
label_embedding (Embedding)     (None, 1, 40)        80          generator_label_input[0][0]
__________________________________________________________________________________________________
generator_input_reshape (Reshap (None, 16, 2048)     0           generator_input_dense[0][0]
__________________________________________________________________________________________________
label_dense (Dense)             (None, 1, 16)        656         label_embedding[0][0]
__________________________________________________________________________________________________
re_lu (ReLU)                    (None, 16, 2048)     0           generator_input_reshape[0][0]
__________________________________________________________________________________________________
label_respahe (Reshape)         (None, 16, 1)        0           label_dense[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 16, 2049)     0           re_lu[0][0]
                                                                 label_respahe[0][0]
__________________________________________________________________________________________________
tf.expand_dims (TFOpLambda)     (None, 1, 16, 2049)  0           concatenate[0][0]
__________________________________________________________________________________________________
generator_Tconv_0 (Conv2DTransp (None, 1, 64, 1024)  52455424    tf.expand_dims[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze (TFOpLambd (None, 64, 1024)     0           generator_Tconv_0[0][0]
__________________________________________________________________________________________________
tf.expand_dims_1 (TFOpLambda)   (None, 1, 64, 1024)  0           tf.compat.v1.squeeze[0][0]
__________________________________________________________________________________________________
generator_Tconv_1 (Conv2DTransp (None, 1, 256, 512)  13107712    tf.expand_dims_1[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze_1 (TFOpLam (None, 256, 512)     0           generator_Tconv_1[0][0]
__________________________________________________________________________________________________
tf.expand_dims_2 (TFOpLambda)   (None, 1, 256, 512)  0           tf.compat.v1.squeeze_1[0][0]
__________________________________________________________________________________________________
generator_Tconv_2 (Conv2DTransp (None, 1, 1024, 256) 3277056     tf.expand_dims_2[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze_2 (TFOpLam (None, 1024, 256)    0           generator_Tconv_2[0][0]
__________________________________________________________________________________________________
tf.expand_dims_3 (TFOpLambda)   (None, 1, 1024, 256) 0           tf.compat.v1.squeeze_2[0][0]
__________________________________________________________________________________________________
generator_Tconv_3 (Conv2DTransp (None, 1, 4096, 128) 819328      tf.expand_dims_3[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze_3 (TFOpLam (None, 4096, 128)    0           generator_Tconv_3[0][0]
__________________________________________________________________________________________________
tf.expand_dims_4 (TFOpLambda)   (None, 1, 4096, 128) 0           tf.compat.v1.squeeze_3[0][0]
__________________________________________________________________________________________________
generator_Tconv_4 (Conv2DTransp (None, 1, 16384, 64) 204864      tf.expand_dims_4[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze_4 (TFOpLam (None, 16384, 64)    0           generator_Tconv_4[0][0]
__________________________________________________________________________________________________
tf.expand_dims_5 (TFOpLambda)   (None, 1, 16384, 64) 0           tf.compat.v1.squeeze_4[0][0]
__________________________________________________________________________________________________
generator_Tconv_5 (Conv2DTransp (None, 1, 114688, 1) 1601        tf.expand_dims_5[0][0]
__________________________________________________________________________________________________
tf.compat.v1.squeeze_5 (TFOpLam (None, 114688, 1)    0           generator_Tconv_5[0][0]
==================================================================================================
Total params: 73,176,289
Trainable params: 73,176,289
Non-trainable params: 0
__________________________________________________________________________________________________
'''

# ------------------------------------------------------------------------------------------------------------------
# 판별자

def discriminator(architecture_size='audio_size',
                  n_classes = 2):
    # generator_filters = [1024, 512, 256, 128, 64] 
    discriminator_filters = [64, 128, 256, 512, 1024, 2048] # 생성자 필터의 역순 + 2048은 flatten 레이어를 위한 것

    
    if architecture_size == 'audio_size':
        audio_input_dim = 114688
        
    label_input = Input(shape=(1,), dtype='int32', name='discriminator_label_input')
    label_em = Embedding(n_classes, n_classes * 20)(label_input) # 양의 정수 (인덱스)를 고정 된 크기의 조밀 한 벡터로 변환합니다.
    label_em = Dense(audio_input_dim)(label_em)
    label_em = Reshape((audio_input_dim, 1))(label_em)

    discriminator_input = Input(shape=(audio_input_dim, 1), name='discriminator_input') # 생성자의 마지막 shape을 넣어준다.
    x = Concatenate()([discriminator_input, label_em]) 

    if architecture_size == 'audio_size':
        
        # layers
        for i in range(4):
            x = Conv1D(
                filters = discriminator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding = 'same'
                , name = f'discriminator_conv_{i}'
                )(x)
            x = LeakyReLU(alpha = 0.2)(x)

        #last 2 layers without phase shuffle
        x = Conv1D(
            filters = discriminator_filters[4]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_4'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv1D(
            filters = discriminator_filters[5]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_5'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
        
    discriminator_output = Dense(1)(x) # 가짜인지 진짜인지 판별
    discriminator = Model([discriminator_input, label_input], discriminator_output, name = 'Discriminator')
    return discriminator

model = discriminator()
model.summary()
'''
Model: "Discriminator"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
discriminator_label_input (Inpu [(None, 1)]          0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 40)        80          discriminator_label_input[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 1, 114688)    4702208     embedding[0][0]
__________________________________________________________________________________________________
discriminator_input (InputLayer [(None, 114688, 1)]  0
__________________________________________________________________________________________________
reshape (Reshape)               (None, 114688, 1)    0           dense[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 114688, 2)    0           discriminator_input[0][0]
                                                                 reshape[0][0]
__________________________________________________________________________________________________
discriminator_conv_0 (Conv1D)   (None, 28672, 64)    3264        concatenate_1[0][0]
__________________________________________________________________________________________________
leaky_re_lu (LeakyReLU)         (None, 28672, 64)    0           discriminator_conv_0[0][0]
__________________________________________________________________________________________________
discriminator_conv_1 (Conv1D)   (None, 7168, 128)    204928      leaky_re_lu[0][0]
__________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)       (None, 7168, 128)    0           discriminator_conv_1[0][0]
__________________________________________________________________________________________________
discriminator_conv_2 (Conv1D)   (None, 1792, 256)    819456      leaky_re_lu_1[0][0]
__________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)       (None, 1792, 256)    0           discriminator_conv_2[0][0]
__________________________________________________________________________________________________
discriminator_conv_3 (Conv1D)   (None, 448, 512)     3277312     leaky_re_lu_2[0][0]
__________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)       (None, 448, 512)     0           discriminator_conv_3[0][0]
__________________________________________________________________________________________________
discriminator_conv_4 (Conv1D)   (None, 112, 1024)    13108224    leaky_re_lu_3[0][0]
__________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)       (None, 112, 1024)    0           discriminator_conv_4[0][0]
__________________________________________________________________________________________________
discriminator_conv_5 (Conv1D)   (None, 28, 2048)     52430848    leaky_re_lu_4[0][0]
__________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)       (None, 28, 2048)     0           discriminator_conv_5[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 57344)        0           leaky_re_lu_5[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            57345       flatten[0][0]
==================================================================================================
Total params: 74,603,665
Trainable params: 74,603,665
Non-trainable params: 0
__________________________________________________________________________________________________

'''

