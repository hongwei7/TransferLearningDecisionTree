from collections import Counter
import numpy as np
import time
import random

class _data(object):#自定义数据集类型
    def __init__(self, x, y):
        self.x = x
        self.y = y


class node(object):#单节点
    def __init__(self, data, index_list):
        self.data = data
        self.clas = Counter(data.y).most_common()[0][0]
        self.index_list = index_list


class condition_node(object):#非单节点
    def __init__(self, f_index, index_list, data, alpha):
        self.alpha = alpha
        self.data = data
        self.f_index = f_index
        self.index_list = index_list
        self.clas = Counter(data.y).most_common()[0][0]

    def create_nodes(self, data):
        self.branches = dict()
        for f in set(data.x.T[self.f_index]):
            self.branches[f] = create_tree(_data(
                data.x[data.x.T[self.f_index] == f], 
                data.y[data.x.T[self.f_index] == f]), 
            self.index_list[:], alpha=self.alpha)


def claculate_H_D(data):#计算信息熵
    H_D = 0
    for yi in set(data.y):
        Ck_D = (data.y == yi).sum() / len(data.y)
        H_D += -Ck_D * np.log(Ck_D)
    return H_D


def claculate_H_D_A(data, A):#计算条件熵
    H_D_A = 0
    for a in set(data.x.T[A]):
        H_D_A += (data.x.T[A] == a).sum() / len(data.y) * claculate_H_D(
            _data(data.x[data.x.T[A] == a], data.y[data.x.T[A] == a]))
    return H_D_A


def claculate_max_g(data, index_list, alpha): #alpha为预剪枝参数
    max_index = index_list[0]
    max_g = 0
    H_D = claculate_H_D(data)
    for index in index_list:
        g_D_index = (H_D - claculate_H_D_A(data, index)) 
        + alpha * (len(Counter(data.x.T[index]).keys()) - 1) #预剪枝
        if max_g < g_D_index:
            max_index = index
            max_g = g_D_index
    if max_g != 0:
        pass
        #print('max_g:', max_g) #显示信息增益具体值
    return max_index, max_g


def create_tree(data, index_list, alpha=0,_e=0.001):
    if len(Counter(data.y)) <= 1 or len(data.y) == 0 or len(index_list) < 1:
        return node(data, index_list)#返回单节点树
    else:
        f_index, f_g = claculate_max_g(data, index_list,alpha)
        if f_g <= _e:
            return node(data, index_list)#返回单节点树
        #print(f_index, index_list) #显示分类条件
        index_list.remove(f_index)
        branch = condition_node(
            f_index=f_index, index_list=index_list, data=data,alpha=alpha)
        branch.create_nodes(data)
        return branch

def predict(root, x):
    result = []
    for xi in x:
        locate = root #locate作为指针
        while(1):
            try:
                locate = locate.branches[xi[locate.f_index]]
                #print(f_index)
            except:
                result.append(locate.clas)
                break
    return result

def spredict(root, xi):
    locate = root #locate作为指针
    while(1):
        try:
            print(locate.f_index)
            locate = locate.branches[xi[locate.f_index]]
        except:
            break

def TDT(T_source,target_Data):
    T_target=T_source
    target_features=list(range(target_Data.x.shape[1]))
    Q=[]
    #print(target_features,(T_source.index_list),T_source.f_index)
    for ai in target_features:
        if(ai not in T_source.index_list and ai!=T_source.f_index):
            Q.append(ai)
    random.shuffle(Q)
    #print(Q)
    for ai in Q:
        for i,xi in  enumerate(target_Data.x):
            locate = T_target #第一轮迁移
            locate_father=T_target
            while(1):
                try:
                    locate.branches[xi[locate.f_index]]#检查是否到头或缺少特征
                    locate_father=locate
                    locate = locate.branches[xi[locate.f_index]]
                except:
                    if(str(type(locate))=="<class '__main__.condition_node'>"):
                        locate.branches[xi[locate.f_index]]=node(_data([xi],[target_Data.y[i]]),[])
                    break
            if locate.clas==target_Data.y[i]:
                pass
            else:
                locate_father.branches[xi[locate_father.f_index]]=condition_node(ai,[],locate_father.branches[xi[locate_father.f_index]].data,0)
                locate_father.branches[xi[locate_father.f_index]].branches=dict()
                locate_father.branches[xi[locate_father.f_index]].branches[xi[ai]]=node(_data([xi],[target_Data.y[i]]),[])
        for i,xi in  enumerate(target_Data.x):
            locate = T_target #第二轮迁移
            while(1):
                try:
                    locate.branches[xi[locate.f_index]]
                    locate_father=locate
                    locate = locate.branches[xi[locate.f_index]]
                except:
                    break
            if locate.clas==target_Data.y[i]:
                pass
            else:
                locate_father.branches[xi[ai]]=node(_data([xi],[target_Data.y[i]]),[])                
    return T_target

def load_data(file_name='flag.data'):
    file = open(file_name)
    r_X,r_y= [],[]
    for line in file.readlines():
        items = str(line).split(',')
        r_y.append(items[6])
        items =items[7:]
        r_X.append(items[1:])
    del r_X[-1], r_y[-1]
    X,y = np.array(r_X).T,np.array(r_y)
    X = X.T
    data = _data(X, y)
    index_list = list(range(data.x.shape[1]))
    return data, index_list
 
def create_source_data(data,index_list):#构造迁移源
    m,n=0,10  #取m到n的特征,m设定不为0时可能有bug(未检验)
    source_data=_data(data.x[:,:n],data.y[:])
    s_index_list=index_list[m:n]
    source_data.x[:,:m]=0
    return source_data,s_index_list,m

def random_some_label(y): #伪随机
    m=len(y)
    domainy=list(set(y))
    for i,yi in enumerate(y):
        if i%3==1:
            y[i]=domainy[i%len(domainy)]
    return y

def main():
    print("Same type of tasks")

    data, index_list = load_data()
    source_data,sindex_list,m=create_source_data(data,index_list)
    target_data, _index_list = load_data()
    print("source data size",(source_data.x.shape[0],source_data.x.shape[1]-m))
    print("target data size",target_data.x.shape)
    
    t1=time.time()
    decision_tree = create_tree(target_data, index_list)
    targetpre_y = predict(decision_tree, target_data.x)
    print('ID3(target) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))
    
    root=condition_node(0,[],target_data,0)
    root.branches=dict()
    TDT_tree =TDT(root ,target_data)
    targetpre_y = predict(TDT_tree, target_data.x)
    print('TDT(target) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))
    
    decision_tree = create_tree(source_data, sindex_list)
    targetpre_y = predict(decision_tree, target_data.x)
    print('ID3(source) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))

    decision_tree = create_tree(source_data, sindex_list)
    T_target=TDT(decision_tree,target_data)
    targetpre_y = predict(T_target, target_data.x)
    print('TDT(transfer) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))
    t2=time.time()
    print('used time:'+str(t2-t1)[:6]+'s')
    print("Distinct but similar tasks:")

    data, index_list = load_data()
    source_data,sindex_list,m=create_source_data(data,index_list)
    target_data, _index_list = load_data()
    target_data.y=random_some_label(target_data.y)
    print("source data size",(source_data.x.shape[0],source_data.x.shape[1]-m))
    print("target data size",target_data.x.shape)
    
    t1=time.time()
    decision_tree = create_tree(target_data, index_list)
    targetpre_y = predict(decision_tree, target_data.x)
    print('ID3(target) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))

    root=condition_node(0,[],target_data,0)
    root.branches=dict()
    TDT_tree =TDT(root ,target_data)
    targetpre_y = predict(TDT_tree, target_data.x)
    print('TDT(target) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))

    decision_tree = create_tree(source_data, sindex_list)
    targetpre_y = predict(decision_tree, target_data.x)
    print('ID3(source) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))

    decision_tree = create_tree(source_data, sindex_list)
    T_target=TDT(decision_tree,target_data)
    targetpre_y = predict(T_target, target_data.x)
    print('TDT(transfer) accuracy:\t',(np.array(targetpre_y) == target_data.y).sum() / len(targetpre_y))
    t2=time.time()
    print('used time:'+str(t2-t1)[:6]+'s')
    

if __name__ == '__main__':
    main()
