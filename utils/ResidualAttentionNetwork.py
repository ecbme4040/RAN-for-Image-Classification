from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, AveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Multiply, Flatten, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
from tensorflow.keras.models import Model

class ResidualAttentionNetwork():
    """
    Implementation of Residual Attention Network for Image Classification.
    This notebook includes the Attention-56, Attention-92 network structure and 
    the building blocks for the network.
    """
    def __init__(self, input_shape, output_size, p=1, t=2, r=1, 
                 filter_dic = {'s1': [16,16,64],
                               's2': [32,32,128],
                               's3': [64,64,256],
                               'se': [128,128,512]}):
        """
        :input_shape: 3 elements tuple, (width, height, channel)
        :output_size: number of categories
        :p: number of residual units in each stage
        :t: number of residual units in trunk branch
        :r: number of residual units in soft mask branch
        :filter_dict: the filter size in each stage
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.p = p
        self.t = t
        self.r = r
        self.filter_dic = filter_dic
      
   
    def Attention_56(self):
        """
        Attention-56 Network Structure: one attention module in each stage,
                                        default learning_mechanism is setting to ARL
        """
        filter_dic = self.filter_dic
        
        input_data = Input(shape=self.input_shape)  #32x32
#         padded_data = ZeroPadding2D((4,4))(input_data)
#         random_crop_layer  = RandomCrop(32, 32)
#         cropped_data = random_crop_layer(padded_data) 
        conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same')(input_data)  #16x16 

        # Residual-Attention Module stage #1 
        filters_s1 = filter_dic['s1']
        res_unit_1 = self.ResidualUnit(conv_1, filters=filters_s1, residual_unit_type='in module')
        am_unit_1 = self.AttentionModuleStage1(res_unit_1, filters=filters_s1, learning_mechanism ='ARL')  #16x16
        
        # Residual-Attention Module stage #2
        filters_s2 = filter_dic['s2']
        res_unit_2 = self.ResidualUnit(am_unit_1, filters=filters_s2, residual_unit_type='out module')
        am_unit_2 = self.AttentionModuleStage2(res_unit_2, filters=filters_s2, learning_mechanism='ARL')  #8x8
      
        # Residual-Attention Module stage #3
        filters_s3 = filter_dic['s3']
        res_unit_3 = self.ResidualUnit(am_unit_2, filters=filters_s3, residual_unit_type='out module')
        am_unit_3 = self.AttentionModuleStage3(res_unit_3, filters=filters_s3, learning_mechanism='ARL')  #4x4

        filters_ending = filter_dic['se']
        am_unit_3 = self.ResidualUnit(am_unit_3, filters=filters_ending, residual_unit_type='out module')  #2x2
        for _ in range(2):
            am_unit_3 = self.ResidualUnit(am_unit_3, filters=filters_ending, residual_unit_type='in module')  #2x2

        batch_norm_2 = BatchNormalization()(am_unit_3)
        activation_2 = Activation('relu')(batch_norm_2)
        avg_pool = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(activation_2)  #1x1
        
        flatten = Flatten()(avg_pool)
        output = Dense(self.output_size, activation='softmax')(flatten)
        
        model = Model(inputs=input_data, outputs=output)
        
        return model
    
    
    def Attention_92(self):
        """
        Attention-92 Network Structure: two attention modules in each stage,
                                        default learning_mechanism is setting to ARL
        """
        filter_dic = self.filter_dic
        
        input_data = Input(shape=self.input_shape)
#         padded_data = ZeroPadding2D((4,4))(input_data)
#         random_crop_layer  = RandomCrop(32, 32)
#         cropped_data = random_crop_layer(padded_data) 
        conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same')(input_data)

        # Residual-Attention Module stage #1 
        filters_s1 = filter_dic['s1']
        res_unit_1 = self.ResidualUnit(conv_1, filters=filters_s1, residual_unit_type='in module')
        am_unit_1 = self.AttentionModuleStage1(res_unit_1, filters=filters_s1, learning_mechanism ='ARL')
        
        # Residual-Attention Module stage #2
        filters_s2 = filter_dic['s2']
        res_unit_2 = self.ResidualUnit(am_unit_1, filters=filters_s2, residual_unit_type='out module')

        am_unit_2 = self.AttentionModuleStage2(res_unit_2, filters=filters_s2, learning_mechanism='ARL')
        am_unit_2 = self.AttentionModuleStage2(am_unit_2, filters=filters_s2, learning_mechanism='ARL')

        # Residual-Attention Module stage #3
        filters_s3 = filter_dic['s3']
        res_unit_3 = self.ResidualUnit(am_unit_2, filters=filters_s3, residual_unit_type='out module')
        am_unit_3 = self.AttentionModuleStage3(res_unit_3, filters=filters_s3, learning_mechanism='ARL')
        am_unit_3 = self.AttentionModuleStage3(am_unit_3, filters=filters_s3, learning_mechanism='ARL')
        am_unit_3 = self.AttentionModuleStage3(am_unit_3, filters=filters_s3, learning_mechanism='ARL')

        filters_ending = filter_dic['se']
        am_unit_3 = self.ResidualUnit(am_unit_3, filters=filters_ending, residual_unit_type='out module')
        for _ in range(2):
            am_unit_3 = self.ResidualUnit(am_unit_3, filters=filters_ending, residual_unit_type='in module')

        batch_norm_2 = BatchNormalization()(am_unit_3)
        activation_2 = Activation('relu')(batch_norm_2)
        avg_pool = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(activation_2)
  
        flatten = Flatten()(avg_pool)
        output = Dense(self.output_size, activation='softmax')(flatten)
        
        model = Model(inputs=input_data, outputs=output)
        
        return model
    
    
    def ResidualUnit(self, residual_input, filters, residual_unit_type='in module'):
        """
        Pre-activation Residual Unit with types: 1. Standard: which do not modify the image size
                                                 2. DownSampling: image is down sampled by a factor of 2
        """
        identity_x = residual_input
        
        filter1, filter2, filter3 = filters
        
        #the 1x1 layers are responsible for reducing and then increasing (restoring) dimensions
        batch_norm_1 = BatchNormalization()(residual_input)
        activation_1 = Activation('relu')(batch_norm_1)
        conv_1 = Conv2D(filters=filter1, kernel_size=(1,1), padding='same')(activation_1)
        
        #the 3x3 layer: a bottleneck with smallerinput/output dimensions.
        batch_norm_2 = BatchNormalization()(conv_1)
        activation_2 = Activation('relu')(batch_norm_2)
        
        #defines the residual unit type
        if residual_unit_type == 'in module':
            conv_2 = Conv2D(filters=filter2, kernel_size=(3,3), strides=(1,1), padding='same')(activation_2)
        else: 
            conv_2 = Conv2D(filters=filter2, kernel_size=(3,3), strides=(2,2), padding='same')(activation_2)

        batch_norm_3 = BatchNormalization()(conv_2)
        activation_3 = Activation('relu')(batch_norm_3)
        conv_3 = Conv2D(filters=filter3, kernel_size=(1,1), padding='same')(activation_3)

        if identity_x.shape != conv_3.shape:
            filter_c = conv_3.shape[-1]
            if residual_unit_type == 'in module':
                identity_x = Conv2D(filters=filter_c, kernel_size=(1,1),strides=(1,1), padding='same')(identity_x) 
            else:  
                identity_x = Conv2D(filters=filter_c, kernel_size=(3,3),strides=(2,2), padding='same')(identity_x) 

        output = Add()([identity_x, conv_3])
        
        return output
    
    
    def AttentionResidualLearning(self, trunk_unit, soft_mask_unit):
        """
        AttentionResidualLearning: ARL
        """
        output = Multiply()([trunk_unit, soft_mask_unit])
        output = Add()([output, trunk_unit])

        return output   
    
    
    def NaiveAttentionLearning(self, trunk_unit, soft_mask_unit):
        """
        NaiveAttentionLearning: NAL
        """
        output = Multiply()([trunk_unit, soft_mask_unit])
        
        return output
        
        
    def AttentionModuleStage1(self, input_unit, filters, learning_mechanism):
        
        for _ in range(self.p):
            am_unit = self.ResidualUnit(input_unit, filters, residual_unit_type='in module')
        
        #trunk branch
        for _ in range(self.t):
            trunk_unit = self.ResidualUnit(am_unit, filters, residual_unit_type='in module')
        
        #soft_mask_branch with 2 skip connections
        ds_unit_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(am_unit)
        for _ in range(self.r):
            ds_unit_1 = self.ResidualUnit(ds_unit_1, filters, residual_unit_type='in module')
        
        skip_unit_outside = self.ResidualUnit(ds_unit_1, filters, residual_unit_type='in module')
        
        ds_unit_2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(ds_unit_1)
        for _ in range(self.r):
            ds_unit_2 = self.ResidualUnit(ds_unit_2, filters, residual_unit_type='in module')
        
        skip_init_inside = self.ResidualUnit(ds_unit_2, filters, residual_unit_type='in module')
        
        ds_unit_3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(ds_unit_2)
        
        for _ in range(self.r * 2):
            ds_unit_3 = self.ResidualUnit(ds_unit_3, filters, residual_unit_type='in module')
        us_unit_1 = UpSampling2D(size=(2,2))(ds_unit_3) 
        
        add_unit_1 = Add()([us_unit_1, skip_init_inside])
        for _ in range(self.r):
            add_unit_1 = self.ResidualUnit(add_unit_1, filters, residual_unit_type='in module')
        us_unit_2 = UpSampling2D(size=(2,2))(add_unit_1) 
        
        add_unit_2 = Add()([us_unit_2, skip_unit_outside])
        for _ in range(self.r):
            add_unit_2 = self.ResidualUnit(add_unit_2, filters, residual_unit_type='in module')
        us_unit_3 = UpSampling2D(size=(2,2))(add_unit_2) 
        
        conv_filter = us_unit_3.shape[-1]
        conv_1 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(us_unit_3)
        conv_2 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(conv_1)
        soft_mask_unit = Activation('sigmoid')(conv_2)
        
        if learning_mechanism == 'NAL':
            output_unit = self.NaiveAttentionLearning(trunk_unit, soft_mask_unit)
        else:
            output_unit = self.AttentionResidualLearning(trunk_unit, soft_mask_unit)
        
        for _ in range(self.p):
            output_unit = self.ResidualUnit(output_unit, filters)
        
        return output_unit
        
    
    def AttentionModuleStage2(self, input_unit, filters, learning_mechanism):
        
        for _ in range(self.p):
            am_unit = self.ResidualUnit(input_unit, filters, residual_unit_type='in module')
        
        #trunk branch
        for _ in range(self.t):
            trunk_unit = self.ResidualUnit(am_unit, filters, residual_unit_type='in module')
        
        #soft_mask_branch with 1 skip connections
        ds_unit_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(am_unit)
        for _ in range(self.r):
            ds_unit_1 = self.ResidualUnit(ds_unit_1, filters, residual_unit_type='in module')

        skip_unit_outside = self.ResidualUnit(ds_unit_1, filters, residual_unit_type='in module')
        
        ds_unit_3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(ds_unit_1)
        for _ in range(self.r * 2):
            ds_unit_3 = self.ResidualUnit(ds_unit_3, filters, residual_unit_type='in module')
        us_unit_1 = UpSampling2D(size=(2,2))(ds_unit_3) 

        add_unit_2 = Add()([us_unit_1, skip_unit_outside])
        for _ in range(self.r):
            add_unit_2 = self.ResidualUnit(add_unit_2, filters, residual_unit_type='in module')
        us_unit_3 = UpSampling2D(size=(2,2))(add_unit_2) 
        
        conv_filter = us_unit_3.shape[-1]
        conv_1 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(us_unit_3)
        conv_2 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(conv_1)
        soft_mask_unit = Activation('sigmoid')(conv_2)

        if learning_mechanism == 'NAL':
            output_unit = self.NaiveAttentionLearning(trunk_unit, soft_mask_unit)
        else:
            output_unit = self.AttentionResidualLearning(trunk_unit, soft_mask_unit)
        
        for _ in range(self.p):
            output_unit = self.ResidualUnit(output_unit, filters)
            
        return output_unit
        
        
    def AttentionModuleStage3(self, input_unit, filters, learning_mechanism):
        
        for _ in range(self.p):
            am_unit = self.ResidualUnit(input_unit, filters, residual_unit_type='in module')
        
        #trunk branch
        for _ in range(self.t):
            trunk_unit = self.ResidualUnit(am_unit, filters, residual_unit_type='in module')
        
        #soft_mask_branch without skip connection
        ds_unit_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(am_unit)
        for _ in range(self.r):
            ds_unit_1 = self.ResidualUnit(ds_unit_1, filters, residual_unit_type='in module')
        us_unit_3 = UpSampling2D(size=(2,2))(ds_unit_1) 
        
        conv_filter = us_unit_3.shape[-1]
        conv_1 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(us_unit_3)
        conv_2 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(conv_1)
        soft_mask_unit = Activation('sigmoid')(conv_2)

        if learning_mechanism == 'NAL':
            output_unit = self.NaiveAttentionLearning(trunk_unit, soft_mask_unit)
        else:
            output_unit = self.AttentionResidualLearning(trunk_unit, soft_mask_unit)
        
        for _ in range(self.p):
            output_unit = self.ResidualUnit(output_unit, filters)
            
        return output_unit
  
        
    def ResNeXtUnit(self, input_unit, filters):
        """
        ResNeXt Unit
        """
        identity_x = residual_input
        filter1, filter2, filter3 = filters
        #[4, 4, 256]

        for i in range(32):
            batch_norm_1 = BatchNormalization()(input_unit)
            activation_1 = Activation('relu')(batch_norm_1)
            conv_1 = Conv2D(filters=filter1, kernel_size=(1,1), padding='same')(activation_1)
            
            batch_norm_2 = BatchNormalization()(conv_1)
            activation_2 = Activation('relu')(batch_norm_2)
            conv_2 = Conv2D(filters=filter2, kernel_size=(3,3), padding='same')(activation_2)
        
            batch_norm_3 = BatchNormalization()(conv_2)
            activation_3 = Activation('relu')(batch_norm_2)
            conv_3 = Conv2D(filters=filter3, kernel_size=(1,1), padding='same')(activation_3)  
            
            if i == 0:
                output = conv_3
        
            else:
                output = Add([output, conv_3])
            output = Add()([conv_3, output])
            
        # shape alignment
        if identity_x.shape[-1] != output.shape[-1]:
            filter_c = output.shape[-1]
            identity_x = Conv2D(filters=filter_c, kernel_size=(1,1), padding='same')(identity_x)
            
        output = Add()([identity_x, output])
        
        return output 