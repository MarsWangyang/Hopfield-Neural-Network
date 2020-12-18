#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tkinter as tk
import math

window = tk.Tk()
window.title('Hopfield Network')
window.geometry('300x400')

#ListBox
#select file
def file_selection():
    value = lb.get(lb.curselection())
    var_selection.set(value)
    raw_data(value)

def sgn(v,b):
    if v - b > 0 :
        return 1
    elif v == b:
        return v
    else:
        return -1

def raw_data(value):
    with open (value ,'r', encoding = 'utf8' ) as f1 :
        data = []
        for line in f1 :                                      #每一行資料的讀取
            x = line.replace(' ','0').replace("\n","")        #first,replace space to 0 and replace \n to space
            for i in range(0,len(str(x))):                    #每一行資料的每一個element做讀取
                real_a = x[i].replace('0','-1')               #next,replace 0 to -1  
                real_b = int(real_a)                          #將string轉為int
                data.append(real_b)                           #最後append to 'data' 就可分開字符
        if value == 'Basic_Training.txt':
            data_1a = np.array([data[0:108]])
            data_2c = np.array([data[108:(108+108)]])
            data_3l = np.array([data[(108*2):(108*3)]])

            data_1a_T = np.transpose(data_1a)
            data_2c_T = np.transpose(data_2c)
            data_3l_T = np.transpose(data_3l)

            data_num = len(data) / 3

            I = np.eye(9*12, dtype = int)

            model_basic(data_1a,data_2c,data_3l,data_1a_T,data_2c_T,data_3l_T,I,data_num)
        elif value == 'Bonus_Training.txt':
            data_1a = np.array([data[0:100]])
            data_2c = np.array([data[100:(100*2)]])
            data_3l = np.array([data[(100*2):(100*3)]])
            data_4  = np.array([data[(100*3):(100*4)]])
            data_5  = np.array([data[(100*4):(100*5)]])
            data_6  = np.array([data[(100*5):(100*6)]])
            data_7  = np.array([data[(100*6):(100*7)]])
            data_8  = np.array([data[(100*7):(100*8)]])
            data_9  = np.array([data[(100*8):(100*9)]])
            data_10  = np.array([data[(100*9):(100*10)]])
            data_11  = np.array([data[(100*10):(100*11)]])
            data_12  = np.array([data[(100*11):(100*12)]])
            data_13  = np.array([data[(100*12):(100*13)]])
            data_14  = np.array([data[(100*13):(100*14)]])
            data_15  = np.array([data[(100*14):(100*15)]])
             
            data_1a_T = np.transpose(data_1a)
            data_2c_T = np.transpose(data_2c)
            data_3l_T = np.transpose(data_3l)
            data_4__T = np.transpose(data_4 )
            data_5__T = np.transpose(data_5 )
            data_6__T = np.transpose(data_6 )
            data_7__T = np.transpose(data_7 )
            data_8__T = np.transpose(data_8 )
            data_9__T = np.transpose(data_9 )
            data_10__T = np.transpose(data_10)
            data_11__T = np.transpose(data_11)
            data_12__T = np.transpose(data_12)
            data_13__T = np.transpose(data_13)
            data_14__T = np.transpose(data_14)
            data_15__T = np.transpose(data_15)
            
            data_num = len(data) / 15

            I = np.eye(10*10, dtype = int)

            model_bonus(data_1a,data_2c,data_3l,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11
                        ,data_12,data_13,data_14,data_15,data_1a_T,data_2c_T,data_3l_T,data_4__T,data_5__T
                        ,data_6__T,data_7__T,data_8__T,data_9__T,data_10__T,data_11__T,data_12__T,data_13__T
                        ,data_14__T,data_15__T,I,data_num)
            
def model_basic(data_1a,data_2c,data_3l,data_1a_T,data_2c_T,data_3l_T,I,data_num):
    w = (((1 / data_num) * 
        (np.dot(data_1a_T,data_1a) + np.dot(data_2c_T,data_2c) + np.dot(data_3l_T,data_3l)))
        - ((3 / data_num) * I))
    bias = []
    for i in range(0,len(w)):                     #bias is the sum of each row 
        for x in range(0,len(w[i])):
            bias_sum = 0
            bias_sum= bias_sum + w[i][x]           
        bias.append(bias_sum)
    
    testing_basic(w,bias,data_1a,data_2c,data_3l)
    
def testing_basic(w,bias,data_1a,data_2c,data_3l):
#------------------remember(asynchronize)---------------------
    with open ('Basic_Testing.txt','r') as f2 :
        data_test = []                                        #讀取測試資料
        data_r = []     #要print測試資料的長相用的
        for line in f2 :                                      #每一行資料的讀取
            data_r.append(line)
            x = line.replace(' ','0').replace("\n","")        #first,replace space to 0 and replace \n to space
            for i in range(0,len(str(x))):                    #每一行資料的每一個element做讀取
                real_a = x[i].replace('0','-1')               #next,replace 0 to -1  
                real_b = int(real_a)                          #將string轉為int
                data_test.append(real_b)                      #最後append to 'data_test' 就可分開字符
        
        data_show = ''.join(data_r)  #測試資料的字串結合
        
        test_1a = np.array(data_test[0:108])                  #test(1,108)
        test_2c = np.array(data_test[108:(108+108)])
        test_3l = np.array(data_test[(108*2):(108*3)])
        
        test_1a_T = np.transpose(test_1a)                     #test_1a_T(108,1)
        test_2c_T = np.transpose(test_2c)
        test_3l_T = np.transpose(test_3l)
        
        test_num = len(data_test) / 3
     
    for cnt1 in range(0,len(w)):                                   #first testing data   
        v = np.dot(w[cnt1],test_1a)     
        y = sgn(v,bias[cnt1])
        test_1a[cnt1] = y     
    
    for cnt2 in range(0,len(w)):                                   #second testing data   
        v = np.dot(w[cnt2],test_2c)     
        y = sgn(v,bias[cnt2])
        test_2c[cnt2] = y 
    
    for cnt3 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt3],test_3l)     
        y = sgn(v,bias[cnt3])
        test_3l[cnt3] = y 
    
    print(" 測試資料： ")
    print(data_show)
    
    final_basic(data_1a,test_1a,data_2c,test_2c,data_3l,test_3l)
        
def final_basic(data_1a,test_1a,data_2c,test_2c,data_3l,test_3l):       #將testing data 整理回圖形狀態
    test_1a_l = list(test_1a)       #為int的list
    test_2c_l = list(test_2c)
    test_3l_l = list(test_3l)
    
    test_1a_list = []              #轉換成str的list
    test_2c_list = []
    test_3l_list = []
    
    n1 = 8
    for a in range(0,len(test_1a_l)):                        #將testing data轉換為圖形
        x1 = str(test_1a_l[a]).replace('-1',' ')             #此行把-1換回space鍵
        test_1a_list.append(x1)                              #並且一個一個的字串輸入至test_xx_list
        if a == n1 :                                          #但如果遇到要換行的時候ex:[9],[19],[29]...
            test_1a_list.append('\n')                        #append a Enter 
            n1 += 9
    test_1a_show = ''.join(test_1a_list)                     #最後將所有字串合併在一起，成為test_xx_show
    
    n2 = 8
    for c in range(0,len(test_2c_l)):    
        x2 = str(test_2c_l[c]).replace('-1',' ')
        test_2c_list.append(x2) 
        if c == n2 :
            test_2c_list.append('\n')
            n2 += 9
    test_2c_show = ''.join(test_2c_list)
    
    n3 = 8
    for l in range(0,len(test_3l_l)):    
        x3 = str(test_3l_l[l]).replace('-1',' ')
        test_3l_list.append(x3) 
        if l == n3 :
            test_3l_list.append('\n')
            n3 += 9
    test_3l_show = ''.join(test_3l_list)

    print('\n',"回想圖形： ")
    print(test_1a_show)
    print(test_2c_show)
    print(test_3l_show)
    
    
        
def model_bonus(data_1a,data_2c,data_3l,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15,data_1a_T,data_2c_T,data_3l_T,data_4__T,data_5__T,data_6__T,data_7__T,data_8__T,data_9__T,data_10__T,data_11__T,data_12__T,data_13__T,data_14__T,data_15__T,I,data_num):
    w = (((1 / data_num) * 
        (np.dot(data_1a_T,data_1a) + np.dot(data_2c_T,data_2c) + np.dot(data_3l_T,data_3l) + np.dot(data_4__T,data_4)
        + np.dot(data_5__T,data_5)+ np.dot(data_6__T,data_6) + np.dot(data_7__T,data_7)
        + np.dot(data_8__T,data_8)+ np.dot(data_9__T,data_9) + np.dot(data_10__T,data_10)
        + np.dot(data_11__T,data_11)+ np.dot(data_12__T,data_12) + np.dot(data_13__T,data_13)
        + np.dot(data_14__T,data_14)+ np.dot(data_15__T,data_15)))
        - ((15 / data_num) * I))
    
    bias = []
    for i in range(0,len(w)):                     #bias is the sum of each row 
        for x in range(0,len(w[i])):
            bias_sum = 0
            bias_sum= bias_sum + w[i][x]           
        bias.append(bias_sum)
    
    testing_bonus(w,bias,data_1a,data_2c,data_3l,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15)
    
def testing_bonus(w,bias,data_1a,data_2c,data_3l,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15):
#------------------remember(asynchronize)---------------------
    with open ('Bonus_Testing.txt','r') as f2 :
        data_test = []                                        #讀取測試資料
        data_r = []     #要print測試資料的長相用的
        for line in f2 :                                      #每一行資料的讀取
            data_r.append(line)
            x = line.replace(' ','0').replace("\n","")        #first,replace space to 0 and replace \n to space
            for i in range(0,len(str(x))):                    #每一行資料的每一個element做讀取
                real_a = x[i].replace('0','-1')               #next,replace 0 to -1  
                real_b = int(real_a)                          #將string轉為int
                data_test.append(real_b)                      #最後append to 'data_test' 就可分開字符
        
        data_show = ''.join(data_r)  #測試資料的字串結合
        
        test_1a = np.array(data_test[0:100])                  #test(1,108)
        test_2c = np.array(data_test[100:(100*2)])
        test_3l = np.array(data_test[(100*2):(100*3)])
        test_4  = np.array(data_test[(100*3):(100*4)])
        test_5  = np.array(data_test[(100*4):(100*5)])
        test_6  = np.array(data_test[(100*5):(100*6)])
        test_7  = np.array(data_test[(100*6):(100*7)])
        test_8  = np.array(data_test[(100*7):(100*8)])
        test_9  = np.array(data_test[(100*8):(100*9)])
        test_10 = np.array(data_test[(100*9):(100*10)])
        test_11 = np.array(data_test[(100*10):(100*11)])
        test_12 = np.array(data_test[(100*11):(100*12)])
        test_13 = np.array(data_test[(100*12):(100*13)])
        test_14 = np.array(data_test[(100*13):(100*14)])
        test_15 = np.array(data_test[(100*14):(100*15)])

        test_num = len(data_test) / 15
        
    for cnt1 in range(0,len(w)):                                   #first testing data   
        v = np.dot(w[cnt1],test_1a)     
        y = sgn(v,bias[cnt1])
        test_1a[cnt1] = y        
    for cnt2 in range(0,len(w)):                                   #second testing data   
        v = np.dot(w[cnt2],test_2c)     
        y = sgn(v,bias[cnt2])
        test_2c[cnt2] = y     
    for cnt3 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt3],test_3l)     
        y = sgn(v,bias[cnt3])
        test_3l[cnt3] = y 
    for cnt4 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt4],test_4)     
        y = sgn(v,bias[cnt4])
        test_4[cnt4] = y     
    for cnt5 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt5],test_5)     
        y = sgn(v,bias[cnt5])
        test_5[cnt5] = y         
    for cnt6 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt6],test_6)     
        y = sgn(v,bias[cnt6])
        test_6[cnt6] = y    
    for cnt7 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt7],test_7)     
        y = sgn(v,bias[cnt7])
        test_7[cnt7] = y 
    for cnt8 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt8],test_8)     
        y = sgn(v,bias[cnt8])
        test_8[cnt8] = y 
    for cnt9 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt9],test_9)     
        y = sgn(v,bias[cnt9])
        test_9[cnt9] = y 
    for cnt10 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt10],test_10)     
        y = sgn(v,bias[cnt10])
        test_10[cnt10] = y 
    for cnt11 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt11],test_11)     
        y = sgn(v,bias[cnt11])
        test_11[cnt11] = y
    for cnt12 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt12],test_12)     
        y = sgn(v,bias[cnt12])
        test_12[cnt12] = y 
    for cnt13 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt13],test_13)     
        y = sgn(v,bias[cnt13])
        test_13[cnt13] = y 
    for cnt14 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt13],test_14)     
        y = sgn(v,bias[cnt14])
        test_14[cnt14] = y 
    for cnt15 in range(0,len(w)):                                   #third testing data   
        v = np.dot(w[cnt15],test_15)     
        y = sgn(v,bias[cnt15])
        test_15[cnt15] = y 
    
    print(" 測試資料： ")
    print(data_show)
    
    final_bonus(data_1a,test_1a,data_2c,test_2c,data_3l,test_3l,data_4,test_4,data_5,test_5,data_6,test_6,data_7,test_7,data_8,test_8,data_9,test_9,data_10,test_10,data_11,test_11,data_12,test_12,data_13,test_13,data_14,test_14,data_15,test_15)
        
def final_bonus(data_1a,test_1a,data_2c,test_2c,data_3l,test_3l,data_4,test_4,data_5,test_5,data_6,test_6,data_7,test_7,data_8,test_8,data_9,test_9,data_10,test_10,data_11,test_11,data_12,test_12,data_13,test_13,data_14,test_14,data_15,test_15):       #將testing data 整理回圖形狀態
    test_1a_l = list(test_1a)       #為int的list
    test_2c_l = list(test_2c)
    test_3l_l = list(test_3l)
    test_4_l = list(test_4)
    test_5_l = list(test_5)
    test_6_l = list(test_6)
    test_7_l = list(test_7)
    test_8_l = list(test_8)
    test_9_l = list(test_9)
    test_10_l = list(test_10)
    test_11_l = list(test_11)
    test_12_l = list(test_12)
    test_13_l = list(test_13)
    test_14_l = list(test_14)
    test_15_l = list(test_15)
    
    test_1a_list = []              #轉換成str的list
    test_2c_list = []
    test_3l_list = []
    test_4_list = []
    test_5_list = []
    test_6_list = []
    test_7_list = []
    test_8_list = []
    test_9_list = []
    test_10_list = []
    test_11_list = []
    test_12_list = []
    test_13_list = []
    test_14_list = []
    test_15_list = []
      
    n1 = 9
    for a in range(0,len(test_1a_l)):                        #將testing data轉換為圖形
        x1 = str(test_1a_l[a]).replace('-1',' ')             #此行把-1換回space鍵
        test_1a_list.append(x1)                              #並且一個一個的字串輸入至test_xx_list
        if a == n1 :                                          #但如果遇到要換行的時候ex:[9],[19],[29]...
            test_1a_list.append('\n')                        #append a Enter 
            n1 += 10
    test_1a_show = ''.join(test_1a_list)                     #最後將所有字串合併在一起，成為test_xx_show
    
    n2 = 9
    for c in range(0,len(test_2c_l)):    
        x2 = str(test_2c_l[c]).replace('-1',' ')
        test_2c_list.append(x2) 
        if c == n2 :
            test_2c_list.append('\n')
            n2 += 10
    test_2c_show = ''.join(test_2c_list)
    
    n3 = 9
    for l in range(0,len(test_3l_l)):    
        x3 = str(test_3l_l[l]).replace('-1',' ')
        test_3l_list.append(x3) 
        if l == n3 :
            test_3l_list.append('\n')
            n3 += 10
    test_3l_show = ''.join(test_3l_list)
    
    n4 = 9
    for d in range(0,len(test_4_l)):    
        x4 = str(test_4_l[d]).replace('-1',' ')
        test_4_list.append(x4) 
        if d == n4 :
            test_4_list.append('\n')
            n4 += 10
    test_4_show = ''.join(test_4_list)
    
    n5 = 9
    for e in range(0,len(test_5_l)):    
        x5 = str(test_5_l[e]).replace('-1',' ')
        test_5_list.append(x5) 
        if e == n5 :
            test_5_list.append('\n')
            n5 += 10
    test_5_show = ''.join(test_5_list)
    
    n6 = 9
    for f in range(0,len(test_6_l)):    
        x6 = str(test_6_l[f]).replace('-1',' ')
        test_6_list.append(x6) 
        if f == n6 :
            test_6_list.append('\n')
            n6 += 10
    test_6_show = ''.join(test_6_list)
    
    n7 = 9
    for g in range(0,len(test_7_l)):    
        x7 = str(test_7_l[g]).replace('-1',' ')
        test_7_list.append(x7) 
        if g == n7 :
            test_7_list.append('\n')
            n7 += 10
    test_7_show = ''.join(test_7_list)
    
    n8 = 9
    for h in range(0,len(test_8_l)):    
        x8 = str(test_8_l[h]).replace('-1',' ')
        test_8_list.append(x8) 
        if h == n8 :
            test_8_list.append('\n')
            n8 += 10
    test_8_show = ''.join(test_8_list)
    
    n9 = 9
    for j in range(0,len(test_9_l)):    
        x9 = str(test_9_l[j]).replace('-1',' ')
        test_9_list.append(x9) 
        if j == n9 :
            test_9_list.append('\n')
            n9 += 10
    test_9_show = ''.join(test_9_list)
    
    n10 = 9
    for k in range(0,len(test_10_l)):    
        x10 = str(test_10_l[k]).replace('-1',' ')
        test_10_list.append(x10) 
        if k == n10 :
            test_10_list.append('\n')
            n10 += 10
    test_10_show = ''.join(test_10_list)
    
    n11 = 9
    for m in range(0,len(test_11_l)):    
        x11 = str(test_11_l[m]).replace('-1',' ')
        test_11_list.append(x11) 
        if m == n11 :
            test_11_list.append('\n')
            n11 += 10
    test_11_show = ''.join(test_11_list)
    
    n12 = 9
    for r in range(0,len(test_12_l)):    
        x12 = str(test_12_l[r]).replace('-1',' ')
        test_12_list.append(x12) 
        if r == n12 :
            test_12_list.append('\n')
            n12 += 10
    test_12_show = ''.join(test_12_list)
    
    n13 = 9
    for o in range(0,len(test_13_l)):    
        x13 = str(test_13_l[o]).replace('-1',' ')
        test_13_list.append(x13) 
        if o == n13 :
            test_13_list.append('\n')
            n13 += 10
    test_13_show = ''.join(test_13_list)
    
    n14 = 9
    for p in range(0,len(test_14_l)):    
        x14 = str(test_14_l[p]).replace('-1',' ')
        test_14_list.append(x14) 
        if p == n14 :
            test_14_list.append('\n')
            n14 += 10
    test_14_show = ''.join(test_14_list)
    
    n15 = 9
    for q in range(0,len(test_15_l)):    
        x15 = str(test_15_l[q]).replace('-1',' ')
        test_15_list.append(x15) 
        if q == n15 :
            test_15_list.append('\n')
            n15 += 10
    test_15_show = ''.join(test_15_list)
    

    print('\n',"回想圖形： ")
    print(test_1a_show)
    print(test_2c_show)
    print(test_3l_show)
    print(test_4_show)
    print(test_5_show)
    print(test_6_show)
    print(test_7_show)
    print(test_8_show)
    print(test_9_show)
    print(test_10_show)
    print(test_11_show)
    print(test_12_show)
    print(test_13_show)
    print(test_14_show)
    print(test_15_show)
    
    
    
#GUI
#Label & Entry
var_selection = tk.StringVar()
L_selection = tk.Label(window , text =  '請選擇您要的檔案：').place(x = 63 , y = 100 )

var_data = tk.StringVar()
var_data.set(('Basic_Training.txt','Bonus_Training.txt'))
lb = tk.Listbox(window , listvariable = var_data , height = 2)
lb.place(x = 65 , y = 120)

#Button
b_go = tk.Button(window , text = 'Go' , bg = 'green' , width = 10 ,height = 8 , command = file_selection )
b_go.place(x = 100 , y = 270 )
window.mainloop()        


# In[ ]:




