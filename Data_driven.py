
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
%matplotlib inline


#### Importing the Data pertaining to Campus Demand:

# In[2]:

dateConverter = lambda d : dt.datetime.strptime(d,'%Y/%m/%d %H:%M:%S')
data = np.genfromtxt('campusDemand.csv',delimiter=",",names=True,dtype=('a255',type(dt),float,),converters={1: dateConverter})


# In[3]:

data


# In[4]:

#Unique meter names, returning the starting indices and counts
name, ind, c  = np.unique(data['Point_name'] , return_index=True,return_counts=True)

#Finding the gaps in the timestamps
fig = plt.figure(figsize=(20,30)) # A 20 inch x 20 inch figure box
for meter,i in zip(name,range(len(name))):
    plt.subplot(4,2,i+1) # 3 rows and 4 columns of subplots
    plt.plot(data[data['Point_name']==meter]['Time'])
    plt.title(meter)
    plt.xlabel('Timestamp')
    plt.ylabel('Timestamp')

#Indices values
k=ind+c-1

# In[5]:

#Unique Meter Name: Duration
for i in range(len(name)):
    print str(name[i])+" from "+str(data[ind[i]]['Time'])+" to "+str(data[ind[i]+counts[i]-1]['Time'])+" : "+str(data[ind[i]+counts[i]-1]['Time']-data[ind[i]]['Time'])
    


# In[6]:

#Finding the gaps in the timestamps of meters
fig = plt.figure(figsize=(20,30)) # A 20 inch x 20 inch figure box
for meter,i in zip(name,range(len(name))):
    plt.subplot(4,2,i+1) # 3 rows and 4 columns of subplots
    plt.plot(data[data['Point_name']==meter]['Time'])
    plt.title(meter)
    plt.xlabel('Timestamp')
    plt.ylabel('Timestamp')


#### Segregating the data based on the meters:
# In[8]:
#Power consumption of all meters in the given data-set:
fig = plt.figure(figsize=(30,40))
for meter,i in zip(name,range(len(name))):
    plt.subplot(7,1,i+1) # 3 rows and 4 columns of subplots
    plt.plot(data[data['Point_name']==meter]['Time'],data[data['Point_name']==meter]['Value'])
    plt.title(meter)
    plt.xlabel('Timestamp')
    plt.ylabel('Power')


### Temperature Data:

# In[9]:

#The temperature data set is renamed to Temp.csv from IW.Weather.Kpit56.Csv.Temp.csv
dateConverter = lambda d : dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S')
temp = np.genfromtxt('Temp.csv',delimiter=",",dtype=(type(dt),float),converters={0: dateConverter},names=['timestamp', 'TempF'], 
                 skiprows=1)


# In[10]:

#Checking for gaps:
time_temp=[temp[i][0] for i in range(len(temp)-1)]
plt.plot(time_temp)
plt.title('Timestamp')


# In[11]:

#15 minute intervals for both time and temperature
temp_new=[]
for i in range(0,len(temp)-1,3): #As there arent any gaps, the loop is run in steps of 3
    temp_new.append(temp[i])
temp=temp_new


### Data Cleansing:

# In[12]:

#Storing timestamps and temperature in 3 variables for the three meters
time_temp3=time_temp2=time_temp0=[temp[i][0] for i in range(len(temp))]
F_temp3=F_temp2=F_temp0=[float(temp[i][1]) for i in range(len(temp))]


#### Equating the timestamps for each meter(Start - End):


# In[15]:

#for all the meters
time_new=[];F_new=[];data_new=['Time','Power','Temperature']
index=[];n=0;index.append(n);count=[]

for j in range(0,len(name)):
    time_new=[]
    F_new=[]
    
    for i in range(0,len(time_temp)-1):
        if(time_temp[i]<data['Time'][k[j]] and time_temp[i]>data['Time'][ind[j]] ):
            #time for each meter in range of temperature
            time_new.append(time_temp[i])
            F_new.append(F_temp[i])
            n=n+1
    index.append(n)

    #Converting timestamp of meter and temperature to a numerical value for interpolation
    Time_meter=[t.minute+t.hour*60+t.day*24*60+t.month*30*24*60+t.year*365*24*60 for t in data['Time'][ind[j]:k[j]]]
    Time_temp=[t.minute+t.hour*60+t.day*24*60+t.month*30*24*60+t.year*365*24*60 for t in time_new]
    clean_power=np.interp(Time_temp,Time_meter,data['Value'][ind[j]:k[j]])

    #Storing all relevant data including timestamp, Power and Temperature for each Meter:
    clean_data=np.vstack((time_new,clean_power,F_new)).T
    data_new=np.vstack((data_new,clean_data))
    count.append(index[j+1]-index[j])





# In[19]:

for i in range(len(name)):
    print "Number of days for "+str( name[i])+" : "+str((data_new[index[i+1]][0]-data_new[index[i]+1][0]).days)


### Near Base and Near peak Load:

# Near Base Load: 2.5$^{th}$ percentile of Daily Load 
# 
# Near Peak Load: 97.5$^{th}$ percentile of Daily Load 

# In[21]:

#Near Base and Near Peak loads for all the meters
q95=[[[] for i in range(1)] for j in range(7)];
q5=[[[] for i in range(1)] for j in range(7)];
Q95=[[[] for i in range(1)] for j in range(7)]
Q5=[[[] for i in range(1)] for j in range(7)]
for j in range(len(name)):
    indices=[0]
    timestamp=[(data_new[t][0]).minute+(data_new[t][0]).hour*60 for t in range(index[j]+1,index[j+1])]
    Max=np.max(timestamp)
    for i in range(count[j]-1):
        if(timestamp[i]==Max): #Which would mean end of the day
            indices.append(i)
    indices.append(index[j+1])
    for k in range(len(indices)-2):
        power=[data_new[i][1] for i in range(index[j]+indices[k]+1,index[j]+indices[k+1]+1)]
        q95[j].append(np.percentile(power, [97.5]))
        q5[j].append(np.percentile(power, [2.5]))



for i in range(7):
    q95[i].remove([])
    q5[i].remove([])


# In[23]:

#Weekdays for each meter readings
weekday=[[[] for i in range(1)] for j in range(7)]
for i in range(7):
    weekday[i]=np.arange(0,len(q95[i]))


# In[24]:

# Plotting the near base and Near Peak Load.
fig = plt.figure(figsize=(30,40)) # A 20 inch x 20 inch figure box
for i in range(len(name)):
    plt.subplot(7,1,i+1) # 3 rows and 4 columns of subplots
    plt.plot(weekday[i],q95[i],'-',label='Near Peak')
    plt.plot(weekday[i],q5[i],'-',label='Near Base')
    plt.title(name[i])
    plt.xlabel('Timestamp')
    plt.ylabel('Power')


## Load Prediction:

### Time of Week:Mondays-Fridays

# In[25]:

#Storing only the data pertaining to Mondays-Fridays
test=[];n=0;index_new=[];
index_new.append(0);
#Meter3
for j in range(len(index)-1):
    for i in range(index[j]+1,index[j+1]):
        if(data_new1[i][0].weekday()!=5 and data_new1[i][0].weekday()!=6):
            test.append(data_new1[i])
            n=n+1;
    index_new.append(n);



### Segregating Occupied and Unoccupied loads:

# In[26]:
#Assuming that the occupied times are in-between 8am - 7pm
testo=[[[] for i in range(1)] for j in range(7)];
testu=[[[] for i in range(1)] for j in range(7)];

for j in range(len(name)):
    for i in range(index_new[j],index_new[j+1]-1):
        if(test[i][0].hour > 8 and test[i][0].hour<19 ):
            testo[j].append(test[i])
        else:
            testu[j].append(test[i])

# In[27]:
#Storing the temp and power corredponding to each meter depending on occupancy:
power_o=[[[] for i in range(1)] for j in range(7)];
temp_o=[[[] for i in range(1)] for j in range(7)];
temp_u=[[[] for i in range(1)] for j in range(7)];
power_u=[[[] for i in range(1)] for j in range(7)];

for i in range(len(testo)-1):
    for j in range(len(testo[i])-1):
        temp_o[i].append(testo[i][j][2])
        power_o[i].append(testo[i][j][1])
    
    for j in range(len(testu[i])-1):
        temp_u[i].append(testu[i][j][2])
        power_u[i].append(testu[i][j][1])


for i in range(7):
    temp_o[i].remove([])
    temp_u[i].remove([])
    power_o[i].remove([])
    power_u[i].remove([])

#Printing the consumption profile based on occupancy times

fig = plt.figure(figsize=(30,40)) # A 20 inch x 20 inch figure box
for i in range(len(name)):
    plt.subplot(7,1,i+1) # 3 rows and 4 columns of subplots
    plt.plot(temp_o[i],power_o[i],'r.',label='Occupied')
    plt.plot(temp_u[i],power_u[i],'b.',label='Unoccupied')
    plt.title(name[i])
    plt.xlabel('Temperature ($^o$F)')
    plt.ylabel('Power in Watts')

### Piece wise Linear Time functions:

# In[30]:

#6 Temperature-intervals for each timestamp between the maximum and minimum
Ti=[[[] for i in range(1)] for j in range(7)];
step=[[[] for i in range(1)] for j in range(7)];

for j in range(len(name)):
    for i in range(index_new[j],index_new[j+1]-1):
        Ti[j].append(test[i][2])
for i in range(7):
    Ti[i].remove([])

for i in range(len(Ti)):
    k=np.linspace(np.min(Ti[i]),np.max(Ti[i]),6)
    step[i].append(k)

for i in range(7):
    step[i].remove([])

print "Temperature-intervals:\n"
for i in range(7):
    print str(name[i])+" : "+str(step[i])


# In[32]:
B=[[[] for i in range(1)] for j in range(7)];
for i in range(len(name)):
    B[i].append(step[i][0][0:5])

for i in range(7):
    B[i].remove([])

print "BOUNDS:\n"
for i in range(7):
    print str(name[i])+" : "+str(B[i])


# In[33]:

#Calculating the Tc's based on the algorithm given in the paper, discussed in report
#Meter3
Tc=np.zeros(6)
indext3=[]
for i in range(0,len(Ti3)):
    if(Ti3[i]<B3[0]):
        Tc[0]=Ti3[i]
    else:
        Tc[0]=B3[0]
        if(Ti3[i]<B3[1]):
            Tc[1]=Ti3[i]-B3[0]
        else:
            Tc[1]=B3[1]-B3[0]
            if(Ti3[i]<B3[2]):
                Tc[2]=Ti3[i]-B3[1]
            else:
                Tc[2]=B3[2]-B3[1]
                if(Ti3[i]<B3[3]):
                    Tc[3]=Ti3[i]-B3[2]
                    
                else:
                    Tc[3]=B3[3]-B3[2]
                    if(Ti3[i]<B3[4]):
                        Tc[4]=Ti3[i]-B3[3]
                    else:
                        Tc[4]=B3[4]-B3[3]
                        Tc[5]=Ti3[i]-B3[4]
                        
    indext3.append(Tc)
    Tc=np.zeros(6)
        


# In[34]:

#Meter2
Tc=np.zeros(6)
indext2=[]
for i in range(0,len(Ti2)):
    if(Ti2[i]<B2[0]):
        Tc[0]=Ti2[i]
    else:
        Tc[0]=B2[0]
        if(Ti2[i]<B2[1]):
            Tc[1]=Ti2[i]-B2[0]
        else:
            Tc[1]=B2[1]-B2[0]
            if(Ti2[i]<B2[2]):
                Tc[2]=Ti2[i]-B2[1]
            else:
                Tc[2]=B2[2]-B2[1]
                if(Ti2[i]<B2[3]):
                    Tc[3]=Ti2[i]-B2[2]
                    
                else:
                    Tc[3]=B2[3]-B2[2]
                    if(Ti2[i]<B2[4]):
                        Tc[4]=Ti2[i]-B2[3]
                    else:
                        Tc[4]=B2[4]-B2[3]
                        Tc[5]=Ti2[i]-B2[4]
                        
    indext2.append(Tc)
    Tc=np.zeros(6)
        
        


# In[35]:

#Meter0
Tc=np.zeros(6)
indext0=[]
for i in range(0,len(Ti0)):
    if(Ti0[i]<B0[0]):
        Tc[0]=Ti0[i]
    else:
        Tc[0]=B0[0]
        if(Ti0[i]<B0[1]):
            Tc[1]=Ti0[i]-B0[0]
        else:
            Tc[1]=B0[1]-B0[0]
            if(Ti0[i]<B0[2]):
                Tc[2]=Ti0[i]-B0[1]
            else:
                Tc[2]=B0[2]-B0[1]
                if(Ti0[i]<B0[3]):
                    Tc[3]=Ti0[i]-B0[2]
                    
                else:
                    Tc[3]=B0[3]-B0[2]
                    if(Ti0[i]<B0[4]):
                        Tc[4]=Ti0[i]-B0[3]
                    else:
                        Tc[4]=B0[4]-B0[3]
                        Tc[5]=Ti0[i]-B0[4]
                        
    indext0.append(Tc)
    Tc=np.zeros(6)
        


### Linear-Regression

#### Considering only occupied state:

# In this case only the occupied condition is considered hence we will have only 486 parameters

# In[36]:

#Values in Matrix A based on the parameters alpha for Regression:
#For one week(480 points)
aa=np.zeros(480)
c3=[];c2=[];c0=[]
for i in range(0,len(aa)):
    aa[i]=1
    c3.append(aa)
    c2.append(aa)
    c0.append(aa)
    aa=np.zeros(480)


# In[37]:

#For the whole time period of the respective meters:
c_3=c3;c_2=c2;c_0=c0
#Meter3
for i in range((len(clean_data_3)-1)/480):
    c_3=np.vstack((c_3,c3))
#Meter2    
for i in range((len(clean_data_2)-1)/480):
    c_2=np.vstack((c_2,c2))
#Meter0    
for i in range((len(clean_data_0)-1)/480):
    c_0=np.vstack((c_0,c0))


# In[38]:

#setting the limit on the final index since the meters dont exactly finish on the 480th point
coeffs3=c_3[0:len(clean_data_3)]
coeffs2=c_2[0:len(clean_data_2)]
coeffs0=c_0[0:len(clean_data_0)]


# In[39]:

A_m3=np.hstack((indext3,coeffs3))
A_m2=np.hstack((indext2,coeffs2))
A_m0=np.hstack((indext0,coeffs0))


# In[41]:

y3=[clean_data_3[i][1] for i in range(len(clean_data_3))]
y2=[clean_data_2[i][1] for i in range(len(clean_data_2))]
y0=[clean_data_0[i][1] for i in range(len(clean_data_0))]


# In[42]:

Y3=np.matrix(y3).T
Y2=np.matrix(y2).T
Y0=np.matrix(y0).T


# In[43]:

print Y3.shape
print Y2.shape
print Y0.shape


# In[44]:

X3 = np.matrix(A_m3)
X2 = np.matrix(A_m2)
X0 = np.matrix(A_m0)


# In[45]:

print X3.shape
print X2.shape
print X0.shape


# In[46]:

# calculate the coefficients
A3=np.linalg.inv(X3.T*X3)*X3.T*Y3
A2=np.linalg.inv(X2.T*X2)*X2.T*Y2
A0=np.linalg.inv(X0.T*X0)*X0.T*Y0


# In[47]:

Load3=X3*A3
Load2=X2*A2
Load0=X0*A0


# In[48]:

print Load3.shape
print Load2.shape
print Load0.shape


# In[49]:

Load_pred3=np.arange(len(time_temp_3))
Load_pred2=np.arange(len(time_temp_2))
Load_pred0=np.arange(len(time_temp_0))

for i in range(0,len(time_temp_3)):
    Load_pred3[i]=Load3[i]
    
for i in range(0,len(time_temp_2)):
    Load_pred2[i]=Load2[i]

for i in range(0,len(time_temp_0)):
    Load_pred0[i]=Load0[i]


# In[50]:

fig1 = plt.figure(figsize=(15,5))
plt.plot(time_temp_3,Load_pred3,'-',label='Predicted Load')
plt.plot(time_temp_3,power_temp_3,'-',label='Actual Load')
plt.title(name[3])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2 = plt.figure(figsize=(15,5))
plt.plot(time_temp_2,Load_pred2,'-',label='Predicted Load')
plt.plot(time_temp_2,power_temp_2,'-',label='Actual Load')
plt.title(name[2])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig3 = plt.figure(figsize=(15,5))
plt.title(name[0])
plt.plot(time_temp_0,Load_pred0,'-',label='Predicted Load')
plt.plot(time_temp_0,power_temp_0,'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[51]:

#A closer look:
fig1 = plt.figure(figsize=(10,5))
plt.title(name[3])
plt.plot(time_temp_3[950:1930],Load_pred3[950:1930],'-',label='Predicted Load')
plt.plot(time_temp_3[950:1930],power_temp_3[950:1930],'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2 = plt.figure(figsize=(10,5))
plt.title(name[0])
plt.plot(time_temp_0[250:1220],Load_pred0[250:1220],'-',label='Predicted Load')
plt.plot(time_temp_0[250:1220],power_temp_0[250:1220],'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig3 = plt.figure(figsize=(10,5))
plt.title(name[2])
plt.plot(time_temp_2[530:1500],Load_pred2[530:1500],'-',label='Predicted Load')
plt.plot(time_temp_2[530:1500],power_temp_2[530:1500],'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[52]:

#Computing the RSS for the above models generated:
rss1_3=np.sqrt(np.sum((Load_pred3-power_temp_3)**2))
rss1_2=np.sqrt(np.sum((Load_pred2-power_temp_2)**2))
rss1_0=np.sqrt(np.sum((Load_pred0-power_temp_0)**2))


# In[53]:

print "RSS values for the respective meters:"
print str(name[3])+" : "+str(rss1_3)
print str(name[2])+" : "+str(rss1_2)
print str(name[0])+" : "+str(rss1_0)


#### Considering only the unoccupied state:

# Using the model for the unoccupied state we get 481 parameters:

# In[54]:

Ti0u=np.matrix(Ti0).T
Ti2u=np.matrix(Ti2).T
Ti3u=np.matrix(Ti3).T


# In[55]:

A_m0u=np.hstack((Ti0u,coeffs0))
A_m2u=np.hstack((Ti2u,coeffs2))
A_m3u=np.hstack((Ti3u,coeffs3))



# In[56]:

X0u = np.matrix(A_m0u)
X2u = np.matrix(A_m2u)
X3u = np.matrix(A_m3u)


# In[57]:

print X3u.shape
print X2u.shape
print X0u.shape


# In[58]:

A0u=np.linalg.inv(X0u.T*X0u)*X0u.T*Y0
A2u=np.linalg.inv(X2u.T*X2u)*X2u.T*Y2
A3u=np.linalg.inv(X3u.T*X3u)*X3u.T*Y3


# In[59]:

Load0u=X0u*A0u
Load2u=X2u*A2u
Load3u=X3u*A3u
#Meter0
Load_pred0u=np.arange(len(time_temp_0))
for i in range(0,len(time_temp_0)):
    Load_pred0u[i]=Load0u[i]
#Meter2
Load_pred2u=np.arange(len(time_temp_2))
for i in range(0,len(time_temp_2)):
    Load_pred2u[i]=Load2u[i]
#Meter3    
Load_pred3u=np.arange(len(time_temp_3))
for i in range(0,len(time_temp_3)):
    Load_pred3u[i]=Load3u[i]


# In[60]:

fig = plt.figure(figsize=(15,5))
plt.plot(time_temp_3,Load_pred3u,'-',label='Predicted Load')
plt.plot(time_temp_3,power_temp_3,'-',label='Actual Load')
plt.title(name[3])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig = plt.figure(figsize=(15,5))
plt.plot(time_temp_0,Load_pred0u,'-',label='Predicted Load')
plt.plot(time_temp_0,power_temp_0,'-',label='Actual Load')
plt.title(name[0])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig = plt.figure(figsize=(15,5))
plt.plot(time_temp_2,Load_pred2u,'-',label='Predicted Load')
plt.plot(time_temp_2,power_temp_2,'-',label='Actual Load')
plt.title(name[2])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



# In[61]:

#Close up (for two weeks)
fig1 = plt.figure(figsize=(10,5))
plt.plot(time_temp_3[950:1930],Load_pred3u[950:1930],'-',label='Predicted Load')
plt.plot(time_temp_3[950:1930],power_temp_3[950:1930],'-',label='Actual Load')
plt.title(name[3])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2 = plt.figure(figsize=(10,5))
plt.plot(time_temp_2[530:1500],Load_pred2u[530:1500],'-',label='Predicted Load')
plt.plot(time_temp_2[530:1500],power_temp_2[530:1500],'-',label='Actual Load')
plt.title(name[2])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig3 = plt.figure(figsize=(10,5))
plt.title(name[0])
plt.plot(time_temp_0[250:1220],Load_pred0u[250:1220],'-',label='Predicted Load')
plt.plot(time_temp_0[250:1220],power_temp_0[250:1220],'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



# In[62]:

rss2_3=np.sqrt(np.sum((Load_pred3u-power_temp_3)**2))
rss2_2=np.sqrt(np.sum((Load_pred2u-power_temp_2)**2))
rss2_0=np.sqrt(np.sum((Load_pred0u-power_temp_0)**2))


# In[63]:

print "RSS values for the respective meters:"
print str(name[3])+" : "+str(rss2_3)
print str(name[2])+" : "+str(rss2_2)
print str(name[0])+" : "+str(rss2_0)


#### Implementing both Occupied and unoccupied conditions:

# Here we take both the conditions, hence will have 487 parameters

# In[64]:

j=np.zeros(6)


# In[65]:

#Piecewise continuity wont apply in case on unoccupied state based on the model:
t3=[]
t2=[];
t0=[];
#Meter3
for i in range(len(clean_data_3)):
    if(clean_data_3[i][0].hour > 8 and clean_data_3[i][0].hour<19 ):
        t3.append(np.hstack((indext3[i],[0])))
    else:
       t3.append(np.hstack((j,Ti3[i]))) 
#Meter2        
for i in range(len(clean_data_2)):
    if(clean_data_2[i][0].hour > 8 and clean_data_2[i][0].hour<19 ):
        t2.append(np.hstack((indext2[i],[0])))
    else:
       t2.append(np.hstack((j,Ti2[i]))) 
#Meter0        
for i in range(len(clean_data_0)):
    if(clean_data_0[i][0].hour > 8 and clean_data_0[i][0].hour<19 ):
        t0.append(np.hstack((indext0[i],[0])))
    else:
       t0.append(np.hstack((j,Ti0[i]))) 


# In[66]:

A_m_3=np.hstack((t3,coeffs3))
A_m_2=np.hstack((t2,coeffs2))
A_m_0=np.hstack((t0,coeffs0))


# In[67]:

X_3 = np.matrix(A_m_3)
X_2 = np.matrix(A_m_2)
X_0 = np.matrix(A_m_0)


# In[68]:

print X_3.shape
print X_2.shape
print X_0.shape


# In[69]:

# Calculating the parameters:
A_3=np.linalg.inv(X_3.T*X_3)*X_3.T*Y3
A_2=np.linalg.inv(X_2.T*X_2)*X_2.T*Y2
A_0=np.linalg.inv(X_0.T*X_0)*X_0.T*Y0


# In[70]:

Load_3=X_3*A_3
Load_2=X_2*A_2
Load_0=X_0*A_0




# In[71]:

Load_pred3=np.arange(len(time_temp_3))
Load_pred2=np.arange(len(time_temp_2))
Load_pred0=np.arange(len(time_temp_0))

for i in range(0,len(time_temp_3)):
    Load_pred3[i]=Load_3[i]
    
for i in range(0,len(time_temp_2)):
    Load_pred2[i]=Load_2[i]

for i in range(0,len(time_temp_0)):
    Load_pred0[i]=Load_0[i]



# In[72]:

fig1 = plt.figure(figsize=(10,5))
plt.plot(time_temp_3,Load_pred3,'-',label='Predicted Load')
plt.plot(time_temp_3,power_temp_3,'-',label='Actual Load')
plt.title(name[3])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2 = plt.figure(figsize=(10,5))
plt.plot(time_temp_2,Load_pred2,'-',label='Predicted Load')
plt.plot(time_temp_2,power_temp_2,'-',label='Actual Load')
plt.title(name[2])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig3 = plt.figure(figsize=(10,5))
plt.title(name[0])
plt.plot(time_temp_0,Load_pred0,'-',label='Predicted Load')
plt.plot(time_temp_0,power_temp_0,'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[73]:

#Close up (for two weeks)
fig1 = plt.figure(figsize=(10,5))
plt.plot(time_temp_3[950:1930],Load_pred3[950:1930],'-',label='Predicted Load')
plt.plot(time_temp_3[950:1930],power_temp_3[950:1930],'-',label='Actual Load')
plt.title(name[3])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2 = plt.figure(figsize=(10,5))
plt.plot(time_temp_2[530:1500],Load_pred2[530:1500],'-',label='Predicted Load')
plt.plot(time_temp_2[530:1500],power_temp_2[530:1500],'-',label='Actual Load')
plt.title(name[2])
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig3 = plt.figure(figsize=(10,5))
plt.title(name[0])
plt.plot(time_temp_0[250:1220],Load_pred0[250:1220],'-',label='Predicted Load')
plt.plot(time_temp_0[250:1220],power_temp_0[250:1220],'-',label='Actual Load')
plt.xlabel('Time')
plt.ylabel('Power in Watts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[74]:

rss3_3=np.sqrt(np.sum((Load_pred3-power_temp_3)**2))
rss3_2=np.sqrt(np.sum((Load_pred2-power_temp_2)**2))
rss3_0=np.sqrt(np.sum((Load_pred0-power_temp_0)**2))


# In[75]:

print "RSS values for the respective meters:"
print str(name[3])+" : "+str(rss3_3)
print str(name[2])+" : "+str(rss3_2)
print str(name[0])+" : "+str(rss3_0)


# In[76]:

Model=["Occupied only","Unoccupied only","Occupied and Unoccupied"]


# In[77]:

rss3=[rss1_3,rss2_3,rss3_3]
rss2=[rss1_2,rss2_2,rss3_2]
rss0=[rss1_0,rss2_0,rss3_0]


# In[78]:

print "Which Model fits best:::"
print str(name[3])+" : "+str(Model[np.argmin(rss3)])
print str(name[2])+" : "+str(Model[np.argmin(rss2)])
print str(name[0])+" : "+str(Model[np.argmin(rss0)])


### Variables Used:

# In[79]:

whos


# In[ ]:



