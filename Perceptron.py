#-*- coding:utf-8 -*-
print ''
class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.weights = [0.0 for i in range(input_num)]
        self.activator = activator
        self.bias = 0.0
    
    def calculate_y(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a,b:a+b,
                map(lambda (x,w):x*w,zip(input_vec,self.weights))
                ,0.0)+self.bias)
    
    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            y = self.calculate_y(input_vec)
            self._update_weights(input_vec, y, label, rate)
    
    def _update_weights(self, input_vec, y, label, rate):
        '''
        按照感知器规则更新权重
        '''
        delta = label - y
        self.weights = map(
            lambda (x,w):w+rate*delta*x,zip(input_vec,self.weights))
        self.bias += rate*delta

    def predict(self, input_vec):
        return self.calculate_y(input_vec)

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\n bias\t:%f' % (self.weights, self.bias)
def f(x):
    return 1 if x>0 else 0

def get_training_dataset():
    input_vecs = [[1,1],[1,0],[0,1],[0,0]]
    labels = [1,0,0,0]
    return input_vecs,labels

def train_and_precptron():
    p = Perceptron(2,f)
    input_vec, labels = get_training_dataset()
    p.train(input_vec, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_and_precptron()
    print and_perceptron
    print '1 and 1 = %d' % and_perceptron.predict([1, 1])
    print '0 and 0 = %d' % and_perceptron.predict([0, 0])
    print '1 and 0 = %d' % and_perceptron.predict([1, 0])
    print '0 and 1 = %d' % and_perceptron.predict([0, 1]) 