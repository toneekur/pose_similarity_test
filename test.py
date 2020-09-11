import numpy as np
import cv2
import ast
import os
from scipy.spatial.distance import cosine,euclidean
import random
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def angle_calculation(first,second, third,fourth):
    first = np.array(first)

    second =np.array(second)
    third = np.array(third)
    fourth = np.array(fourth)


    v1 = second-first
    v2 = fourth-third


    cs = (cosine(v1, v2))
    eu = euclidean(v1,v2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    angle = np.arccos(cosine_angle)
    return angle

img_angles={}
img_c ={}
with open('./keps.txt','r+') as f:
    text = f.readlines()
    for line in text:

        img_path,cc = line.split('=')
        #img = cv2.imread(img_path)

        coords_ = ast.literal_eval(cc)
        if -1 in coords_:
            continue


        coords =np.array(coords_).reshape((18,2))
        #angle = cosine()
        angle1 = angle_calculation(coords[2],coords[3],coords[3],coords[4])
        angle2 = angle_calculation(coords[1], coords[2], coords[2], coords[3])
        angle3 = angle_calculation(coords[5], coords[6], coords[6], coords[7])
        angle4 = angle_calculation(coords[1], coords[5], coords[5], coords[6])
        angle5 = angle_calculation(coords[5], coords[2], coords[5], coords[4])
        angle6 = angle_calculation(coords[8], coords[9], coords[9], coords[10])
        angle7 = angle_calculation(coords[11], coords[12], coords[12], coords[13])
        angle8 = angle_calculation(coords[8], coords[9], coords[8], coords[11])
        angle9 = angle_calculation(coords[11], coords[12], coords[11], coords[8])
        angle10 = angle_calculation(coords[1], coords[2], coords[1], coords[8])
        angle11= angle_calculation(coords[1], coords[5], coords[1], coords[11])
        angle12 = angle_calculation(coords[1], coords[8], coords[8], coords[9])
        angle13 = angle_calculation(coords[1], coords[11], coords[11], coords[12])

        all_angles = [angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10,angle11,angle12,angle13]

        #print (img_path,all_angles)
        img_angles[int(os.path.basename(img_path)[:-4])] = all_angles
        img_c[int(os.path.basename(img_path)[:-4])] = coords_


        #print (angle)

        # for i,c in enumerate(coords):
        #     cv2.circle(img,(c[0],c[1]),5,(100,255,100))
        #     #print (i)
        # cv2.imshow('asd',img)
        # cv2.waitKey(0)
img_angles = dict(sorted(img_angles.items()))
dataset ={}

dataset_test ={}
y =[]
for q in range(len(img_angles.keys())-1):

    rand_ind_1 = random.randint(0,len(img_angles.keys())-1)
    rand_ind_2 = random.randint(0, len(img_angles.keys())-1)

    d1_name = list(img_angles.keys())[rand_ind_1]
    d2_name =list(img_angles.keys())[rand_ind_2]


    a1 = np.array(img_angles[d1_name])
    a2 =np.array(img_angles[d2_name])

    #print (list(img_angles.keys())[q],list(img_angles.keys())[q+1], np.sum(np.abs(a1-a2)))
    #print(f"{d1_name}_{d2_name}--({cosine(a1,a2)})----({np.sum(np.abs(a1-a2))})")
    y.append(1-cosine(a1,a2))
    dataset[f"{d1_name}_{d2_name}"] =np.abs(a1-a2)

#print (dataset)
#y = np.array([0.97,0.6,0.8,0.5,0.8,0.6,0.05,0.1,0.9,0.85])
#y = np.array([1,0.5,1,1,1,1,0,0,1,1])


y =np.array(y).reshape((307,-1))


x = np.array(list(dataset.values()))



scalerx = StandardScaler()
scalery = StandardScaler()

x_sc =scalerx.fit(x)
y_sc = scalery.fit(y)
#print (y_sc.data_max_,y_sc.data_min_,y_sc.data_range_)

new_x = x_sc.transform(x)
new_y = y_sc.transform(y)
#print (y.reshape((1,-1)))

x_tensor = torch.from_numpy(new_x).float()
y_tensor =torch.from_numpy(new_y).float()

#print(x_tensor.shape,y_tensor.shape)
print (x_tensor.shape , y_tensor.shape)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Sequential(
        torch.nn.Linear(13, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 150),
        torch.nn.ReLU(),
        torch.nn.Linear(150,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1))

    def forward(self, x):
        y_pred = self.linear(x)
        #y_pred = torch.sigmoid(y_pred)
        return y_pred
# model = torch.nn.Sequential(
#         torch.nn.Linear(13, 169),
#         torch.nn.ReLU(),
#         torch.nn.Linear(169, 338),
#         torch.nn.ReLU(),
#         torch.nn.Linear(338,100),
#         torch.nn.ReLU(),
#         torch.nn.Linear(100, 50),
#         torch.nn.ReLU(),
#         torch.nn.Linear(50, 1),
#
#         )
model = Model()

if torch.cuda.is_available():
    model = model.cuda()
    X_train_torch, y_train_torch = x_tensor.cuda(), y_tensor.cuda()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)



#print (X_train_torch.size(),y_train_torch.size())
for epoch in range(150):
    output = model.forward(X_train_torch)
    print (output[:5],y_train_torch[:5])
    loss = criterion(output, y_train_torch)
    print('Epoch: ', epoch, 'Loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

