import tensorflow as tf 
import numpy as np 
import os 

# util method , quantile regression loss function , p should be an Integer between 1 to 99
def quan_reg_loss(_label,_output,p):
    r = _label-_output
    p = p/100.0
    return tf.reduce_mean(p * tf.nn.relu(r) + (1-p) * tf.nn.relu(-r))

# get a simple rnn cell: units is the <output dim> and mode is to controll which
# rnn cell you can use
def get_rnn_cell(units,name,mode="LSTM"):
    mode = mode.upper()
    if mode =="LSTM":
        return tf.nn.rnn_cell.BasicLSTMCell(units,forget_bias=1.0,name=name)
    elif mode =="GRU":
        return tf.nn.rnn_cell.GRUCell(units,name=name,kernel_initializer=tf.orthogonal_initializer)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(units,name=name)

class cdf_reg_model:
    def __init__(self,isTraining,save_path="./ckpt/tmp/"):
        tf.reset_default_graph()
        self.isTraining=isTraining
        self.x = None
        self.y = None
        self.save_path = save_path
        # self.anchor is a sorted list which elements are Integer between 1 to 99 ,for instance [25,50,75]
        # for quantile
        self.anchor = [10,30,50,70,90]
        self.pred = None
        self.prob_w = None
        self.quan_out = None
        self.loss = 0.0
        self.sess = None
        self.config = None
        self.saver = None
        self.train_op = None
        self.loss_record = []
        #init
        self._build_network()
        if isTraining==True:
            self.train_init()
        else:
            self.generate_session()
            self._get_saver()
            self.init_variables(saved=True)

    def generate_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        return  

    def _get_saver(self):
        self.saver =  tf.train.Saver()

    def _build_network(self,n_seq=42,input_dim=6):
        self.x = tf.placeholder(tf.float32,shape=[None,n_seq,input_dim])
        self.y = tf.placeholder(tf.float32,shape=[None,1])
        net = self._bigru_feature_engineer()

        #self.prob_w = [min(e,100-e) for e in self.anchor]
        #self.prob_w[len(self.prob_w)//2] += 20 

        self.prob_w= list(np.exp(-np.square(np.array(self.anchor)-50)/500))#500 is a magic number

        self.loss = 0.0
        self.quan_out = []
        for i in range(len(self.anchor)):
        qout = tf.layers.dense(net,1,name="qout"+str(self.anchor[i]))
        self.quan_out.append(qout)
        self.loss += quan_reg_loss(self.y,qout,self.anchor[i])

        self._predict()
        return self.loss

    def _lstm_feature_engineer(self):
        lstmcell1 = get_rnn_cell(units=128,name="lstmcell1",mode="LSTM")
        rnn_out, rnn_final_state = tf.nn.dynamic_rnn(lstmcell1,self.x,dtype=tf.float32)

    #*** note that lstm_final_state is tuple, not a single tensor***
        net = tf.layers.dense(rnn_final_state.h,128,tf.nn.relu,name="dense1")
        return net

    def _gru_feature_engineer(self):
        gru_cell = get_rnn_cell(units=128,name="gru_cell_1",mode="GRU")
        output, hidden_state = tf.nn.dynamic_rnn(gru_cell, inputs=self.x,dtype=tf.float32)
        net = tf.layers.dense(hidden_state,128,tf.nn.relu,name="dense1")
        return net

    def _bigru_feature_engineer(self):
        gru_c1 = get_rnn_cell(units=128,name="gru_cell_1",mode="GRU")
        gru_c2 = get_rnn_cell(units=128,name="gru_cell_2",mode="GRU")
        output, hidden_state_1 = tf.nn.dynamic_rnn(gru_c1, inputs=self.x,dtype=tf.float32)
        output, hidden_state_2 = tf.nn.dynamic_rnn(gru_c2, inputs=tf.reverse(self.x,axis=[1]),dtype=tf.float32)
        merge_tensor = tf.concat([hidden_state_1,hidden_state_2],axis=1)
        net = tf.layers.dense(merge_tensor,128,tf.nn.relu,name="dense1")
        return net

    # use quantile result to get prediction
    def _predict(self):
        tmp = []
        for i in range(len(self.quan_out)):
            tmp.append(self.quan_out[i]*self.prob_w[i])
        self.pred = tf.reduce_sum(tmp,axis=0)/sum(self.prob_w)
        return self.pred

    def init_variables(self,saved=False):
        if saved==False:
            self.sess.run(tf.global_variables_initializer())
            print("global_variables_initialize!!!")
        else:
            try:
                model_file = tf.train.latest_checkpoint(self.save_path)
                self.saver.restore(self.sess,model_file)
                print("The saved model has been restored!!!!")
            except:
                raise IOError("IO failed! \n **Check your save path! ! !**")
        return


    def train_init(self):
        self.generate_session()
        self._get_saver()
        learning_rate = 1e-4
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.init_variables(False)
        return 

    def fit(self,x,y,repeat=1):
        """
        x is ndarray ,shape=[batch,n_seq,feature_dim]
        y is ndarray ,shape=[None,1] regression model
        No output
        """
        if self.isTraining==False:
        print("You can't train this model!")
        return 
        else:
        fetches = [self.loss,self.train_op]
        for i in range(repeat):
            loss,_op = self.sess.run(fetches,{self.x:x, self.y:y})
            self.loss_record.append(loss)
        return loss

    def validate(self,x,y):
        """
        only use while trainning , only output the LOSS of the validation data
        x,y: ndarray, shape is same to  self.fit()
        """
        if self.isTraining == False:
            print("You can't vaildate model while not trainning !")
            return 
        loss = self.sess.run(self.loss,{self.x:x,self.y:y})
        return loss

    def predict(self,x):
        # predict a value with input data x
        res =  self.sess.run(self.pred,{self.x:x})
        return res


    def save_model(self,epoch=0):
        if self.isTraining==False:
            print("You can't save model!")
        else:
            self.saver.save(self.sess,self.save_path+"model.ckpt",global_step=epoch)
            print("model has successfully been saved!")


