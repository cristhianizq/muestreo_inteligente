#!/usr/bin/env python
# coding: utf-8

# # Muestreo Centros Digitales


# In[1]:


import pandas as pd
import numpy as np
from datetime import date
import datetime
import random
import time

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:

import csv
from paramiko import SSHClient
import paramiko
import os
import sys
import NotificateModulePython as notify


## Parametros iniciales
Hostname='100.123.26.248'
Username='DATACENTERDHS/workflowadmin'
Password='Colombia2017*'
PathHab= 'F:/Workflow/Projects/MinTIC_SpeedTest/ProcessScripts/initialize-general-list-hab.csv'
PathFds= 'F:/Workflow/Projects/MinTIC_SpeedTest/ProcessScripts/initialize-general-list-fds.csv'
ubicacionArchivoFestivos="F:/Workflow/Projects/MinTIC_SpeedTest/ProcessScripts/initialize-general-list-holidays.csv"
SqlFile='F:/Workflow/Projects/MinTIC_SpeedTest/ProcessScripts/query-Muestras-SQL.csv'

todayNotify= datetime.datetime.today().strftime('%Y/%m/%d')
dayNotify= datetime.datetime.today().strftime('%A')

dayHabNotify=['Monday','Tuesday','Wednesday','Thursday','Friday']
dayFdsNotify=['Saturday','Sunday']

with open (ubicacionArchivoFestivos, newline='') as csvfile:
        next(csvfile)
        CRCreader= csv.reader(csvfile, delimiter= ';')
        ResultNotify = []
        for row in CRCreader:
            newrow= "/".join(row)
            ResultNotify.append(newrow)

class SSH:
    def __init__(self):
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=Hostname,username=Username,password=Password)
        print("Conexión Exitosa")

    '''def exec_cmd(self,cmd):
        
        stdin,stdout,stderr = self.ssh.exec_command(cmd)
        output=""
        if stderr.channel.recv_exit_status() != 0:
            stdout=stderr.readlines()
        else:
            stdout=stdout.readlines()
            if stdout == []:
                stdout=stderr.readlines()                       
        for line in stdout:
            output=output+line
        
        if output!="":
            print (output)
        else:
            print ("Ejecución Exitosa")'''    
        

    def stpf(self,LocalPath,RemotePath):
        sftp = self.ssh.open_sftp()
        sftp.put(LocalPath, RemotePath, callback=None, confirm=True)
        sftp.close()
        print("Archivo Copiado")


# ## Parámetro para definir el medio de carga de datos

# In[4]:


formas_de_ingreso=['db','archivos']
modoIngresoInformacion=formas_de_ingreso[0] #cambia a 1 si se usan archivos

# In[5]:
con_graficas=False
num_horas=15
pc_muestra=0.3
n_beams_sat=18


# ## Lectura Bases de datos

# In[6]:


if modoIngresoInformacion== 'db':
    ## Llamado funcion para consulta y parametros
    """
    from get_data_from_db import *    
    registros, aps_operativos, num_aps, num_muestras_hora=GetDataFromDB()
    str_list=["ID","BANDA","SUBIDA_LIM","BAJADA_LIM","FECHA","HORA","CONDICION","VELOCIDAD_BAJADA","VELOCIDAD_SUBIDA","CUMPLE_BANDA","CUMPLE_SUBIDA","CUMPLE_BAJADA",'BASE','FASE']
    Data=pd.DataFrame.from_records(registros, columns=str_list)
    """
    #Carga Archivos Prueba
    Data = pd.read_csv(f'query-Muestras-SQL.csv', sep=';')
    num_aps=Data['ID'].nunique()    
    ### Cambio fase4, conservar número de Beam y calculo numero muestras
    num_muestras_hora=math.ceil(num_aps*pc_muestra)
    Data['BEAM']=Data['BASE']
    ### fin Cambio fase4
    Data['BASE']='principal'
    print(num_aps,num_muestras_hora)


# ## Lectura Archivos
# ###### Codigo para realizar el proceso con lectura de archivos

# ### Leer Archivo Festivos

# In[7]:


ubicacionArchivoFestivos="F:/Workflow/Projects\MinTIC_SpeedTest/ProcessScripts/"
ubicacionArchivoFestivos="" #en caso de no ejecutar en el servidor
diasFestivos=pd.read_csv(f'{ubicacionArchivoFestivos}initialize-general-list-holidays.csv',sep=";")
diasFestivos['fecha']=pd.to_datetime(diasFestivos[['YEAR', 'MONTH', 'DAY']])


# ### Preprocesamiento

# In[8]:


Data['FECHA']=pd.to_datetime(Data['FECHA']).dt.date
Data['FECHA']=pd.to_datetime(Data['FECHA'])
maxFecha=Data['FECHA'].max()
minFecha=maxFecha + datetime.timedelta(days=-14)
minFechaFDS=maxFecha + datetime.timedelta(days=-28)


# In[9]:


Data['DIA_S']=Data['FECHA'].dt.dayofweek
Data['FILTRO']=Data['DIA_S'].isin([5,6])
Data['FILTRO']=np.where(Data['DIA_S'].isin([5,6]) , 'fds','hab')
Data['FILTRO']=np.where(Data['FECHA'].isin(diasFestivos['fecha']),'fds',Data['FILTRO'])


# In[10]:


Data.drop_duplicates(subset=["ID","BANDA","SUBIDA_LIM","BAJADA_LIM","FECHA","HORA","CONDICION","VELOCIDAD_BAJADA","VELOCIDAD_SUBIDA","CUMPLE_BANDA","CUMPLE_SUBIDA","CUMPLE_BAJADA"],keep='first', inplace=True)
Data=Data[(Data['HORA']>=6)&(Data['HORA']<=20)]
DataTotal=Data.copy()
Data=Data[(Data['FECHA']>=minFecha)&(Data['FECHA']<maxFecha)]


# ##  Resumenes validación

# In[11]:


def pieChart(col):
    A = col.value_counts()
    indices = A.index
    plt.pie(A, labels=indices, autopct="%0.0f %%", colors=['dodgerblue','red'])
    plt.axis("equal")
    plt.show()    


# In[12]:


## TORTA 1
Data[Data['BASE']=='principal']['CONDICION'].value_counts()


# In[13]:


Data[Data['BASE']=='principal']['CONDICION'].value_counts()/len(Data[Data['BASE']=='principal'])


# In[14]:


if con_graficas:
    pieChart(Data[Data['BASE']=='principal']['CONDICION'])


# In[15]:


resumen_exito_beam=pd.pivot_table(Data[Data['BASE']=='principal'][['BEAM','CONDICION']].value_counts().to_frame().reset_index(), values=0, index=['BEAM'], columns=['CONDICION'], aggfunc=np.sum)
resumen_exito_beam


# In[16]:


# Fase 4,  grafico stack por beam
if con_graficas:
    f, ax = plt.subplots(1,2, figsize=(10,4))
    resumen_exito_beam.plot.bar(stacked=True, ax=ax[0])
    resumen_exito_beam.div(resumen_exito_beam.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax[1])


# In[17]:


DataExito=Data[Data['CONDICION']=='E']


# In[18]:


## TORTA 2
DataExito[DataExito['BASE']=='principal']['CUMPLE_BANDA'].value_counts()


# In[19]:


DataExito[DataExito['BASE']=='principal']['CUMPLE_BANDA'].value_counts()/len(DataExito)


# In[20]:


if con_graficas:
    pieChart(DataExito[DataExito['BASE']=='principal']['CUMPLE_BANDA'])


# In[21]:


resumen_cumple_beam=pd.pivot_table(DataExito[DataExito['BASE']=='principal'][['BEAM','CUMPLE_BANDA']].value_counts().to_frame().reset_index(), values=0, index=['BEAM'], columns=['CUMPLE_BANDA'], aggfunc=np.sum)
resumen_cumple_beam


# In[22]:


# Fase 4,  grafico stack por beam
if con_graficas:
    f, ax = plt.subplots(1,2, figsize=(10,4))
    resumen_cumple_beam.plot.bar(stacked=True, ax=ax[0])
    resumen_cumple_beam.div(resumen_cumple_beam.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax[1])


# In[23]:


DataExito[DataExito['BASE']=='principal']['CUMPLE_SUBIDA'].value_counts()/len(DataExito[DataExito['BASE']=='principal'])


# In[24]:


if con_graficas:
    resumen_cumple_beam=pd.pivot_table(DataExito[DataExito['BASE']=='principal'][['BEAM','CUMPLE_SUBIDA']].value_counts().to_frame().reset_index(), values=0, index=['BEAM'], columns=['CUMPLE_SUBIDA'], aggfunc=np.sum)
    resumen_cumple_beam
    f, ax = plt.subplots(1,2, figsize=(10,4))
    resumen_cumple_beam.plot.bar(stacked=True, ax=ax[0])
    resumen_cumple_beam.div(resumen_cumple_beam.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax[1])


# In[25]:


DataExito[DataExito['BASE']=='principal']['CUMPLE_BAJADA'].value_counts()/len(DataExito[DataExito['BASE']=='principal'])


# In[26]:


if con_graficas:
    resumen_cumple_beam=pd.pivot_table(DataExito[DataExito['BASE']=='principal'][['BEAM','CUMPLE_BAJADA']].value_counts().to_frame().reset_index(), values=0, index=['BEAM'], columns=['CUMPLE_BAJADA'], aggfunc=np.sum)
    resumen_cumple_beam
    f, ax = plt.subplots(1,2, figsize=(10,4))
    resumen_cumple_beam.plot.bar(stacked=True, ax=ax[0])
    resumen_cumple_beam.div(resumen_cumple_beam.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax[1])


# In[27]:


#conteo centros con exito
DataExito[DataExito['BASE']=='principal']['ID'].nunique()


# In[28]:


#conteo centrol críticos
num_aps-DataExito[DataExito['BASE']=='principal']['ID'].nunique()


# In[29]:


###  GRAFICO


# In[30]:


res_hora_cumple_Banda=pd.crosstab(index=DataExito['HORA'],
            columns=DataExito['CUMPLE_BANDA'], margins=True)


# In[31]:


res_hora_cumple_Banda['NO']=res_hora_cumple_Banda['NO']/res_hora_cumple_Banda['All']
res_hora_cumple_Banda['SI']=res_hora_cumple_Banda['SI']/res_hora_cumple_Banda['All']


# In[32]:


pd.crosstab(index=DataExito['HORA'],
            columns=DataExito['CUMPLE_BANDA'], margins=True, normalize='index')


# #### Gráfico

# In[33]:


if con_graficas:
    res_grafico=pd.pivot_table(Data[Data['BASE']=='principal'].groupby(['BANDA','HORA','CUMPLE_BANDA'])['ID'].count().reset_index(), index=['BANDA','HORA'],columns=['CUMPLE_BANDA'], values='ID', aggfunc=np.sum, fill_value=0)
    res_grafico['TOTAL']=res_grafico.sum(axis=1)
    res_grafico['PORCF']=res_grafico['SI']/res_grafico['TOTAL']
    res_grafico.reset_index(inplace=True)
    if con_graficas:
        fig, ax = plt.subplots()
        for label, grp in res_grafico.groupby('BANDA'):
            grp.plot(x = 'HORA', y = 'PORCF',ax = ax, label = label)
    res_grafico.to_csv('grafica.csv')


# In[34]:


if con_graficas:
    res_grafico=pd.pivot_table(DataExito[DataExito['BASE']=='principal'].groupby(['BANDA','HORA','CUMPLE_BANDA'])['ID'].count().reset_index(), index=['BANDA','HORA'],columns=['CUMPLE_BANDA'], values='ID', aggfunc=np.sum, fill_value=0)
    res_grafico['TOTAL']=res_grafico.sum(axis=1)
    res_grafico['PORCF']=res_grafico['SI']/res_grafico['TOTAL']
    res_grafico.reset_index(inplace=True)
    if con_graficas:
        fig, ax = plt.subplots()
        for label, grp in res_grafico.groupby('BANDA'):
            grp.plot(x = 'HORA', y = 'PORCF',ax = ax, label = label)
    res_grafico.to_csv('grafica.csv')


# In[35]:


def quantiles_adaptados(x):
    conteo=x.count()
    if conteo<=5:
        return x.quantile(0.5)
    elif conteo<=10:
        return x.quantile(0.2)
    else:
        return x.quantile(0.1)


# ### Archivos de Resumen

# In[36]:


today = date.today()
D=today.day
M=today.month
Y=today.year


# # Habiles

# In[37]:


Data_Habil= Data[(Data['BASE']=='principal')&(Data['FILTRO']=='hab')].groupby(['ID','HORA']).agg(
                CONTEO=('ID','count'),
                SUBIDA=('VELOCIDAD_SUBIDA', quantiles_adaptados),
                BAJADA=('VELOCIDAD_BAJADA', quantiles_adaptados)
            )


# In[38]:


Data_Habil.reset_index(inplace=True)
BANDA=Data[['ID','BANDA','SUBIDA_LIM','BAJADA_LIM']].drop_duplicates(subset='ID',keep='last')
Data_Habil=Data_Habil.merge(BANDA,left_on='ID',right_on='ID', how='left')
Data_Habil.rename(columns={'BAJADA_LIM':'BAJA','SUBIDA_LIM':'SUBE','BAJADA':'BAJA_R','SUBIDA':'SUBE_R'}, inplace=True)
Data_Habil=Data_Habil[['ID','BANDA','HORA','BAJA','SUBE','BAJA_R','SUBE_R']]


# In[39]:


Data_Habil['CUMPLE_BAJA']=np.where(Data_Habil['BAJA_R']>=Data_Habil['BAJA'],'SI','NO')
Data_Habil['CUMPLE_SUBE']=np.where(Data_Habil['SUBE_R']>=Data_Habil['SUBE'],'SI','NO')
Data_Habil['CUMPLE']=np.where( (Data_Habil['CUMPLE_BAJA']=='SI') & (Data_Habil['CUMPLE_SUBE']=='SI'),'SI','NO')
Data_Habil['PORC_BAJADA']=Data_Habil['BAJA_R']/Data_Habil['BAJA']
Data_Habil['PORC_SUBE']=Data_Habil['SUBE_R']/Data_Habil['SUBE']
Data_Habil['CERCANO_BAJA']=np.where(Data_Habil['PORC_BAJADA']>1, 'CUMPLE', np.where(Data_Habil['PORC_BAJADA']>0.9,'SI','NO'))
Data_Habil['CERCANO_SUBE']=np.where(Data_Habil['PORC_SUBE']>1, 'CUMPLE', np.where(Data_Habil['PORC_SUBE']>0.9,'SI','NO'))
Data_Habil['CERCANO']=np.where( (Data_Habil['CERCANO_BAJA']=='CUMPLE') & (Data_Habil['CERCANO_SUBE']=='CUMPLE'),'CUMPLE',
                                 np.where((Data_Habil['CERCANO_BAJA']=='SI') & (Data_Habil['CERCANO_SUBE']=='SI'),'AMBOS',
                                         np.where((Data_Habil['CERCANO_BAJA']=='CUMPLE') & (Data_Habil['CERCANO_SUBE']=='SI'),'SUBE',
                                                 np.where((Data_Habil['CERCANO_BAJA']=='SI') & (Data_Habil['CERCANO_SUBE']=='CUMPLE'),'BAJA','NO'))))


# In[40]:


Data_Habil_Res=Data_Habil.copy()
Data_Habil_Res['TIPO2']=np.where(Data_Habil_Res['CERCANO']=="NO",'NO','SI')
Data_Habil_Res=pd.pivot_table(Data_Habil_Res[['ID','BANDA','CUMPLE']], index=['ID','BANDA'], columns=['CUMPLE'], aggfunc=len, fill_value=0)
Data_Habil_Res['PORC']=Data_Habil_Res['NO']/(Data_Habil_Res['NO']+Data_Habil_Res['SI'])
Data_Habil_Res['TIPO']= np.where(Data_Habil_Res['PORC']==1,'CRITICO',np.where(Data_Habil_Res['PORC']==0,'BUENO','REGULAR'))


# In[41]:


if con_graficas:
    Data_Habil.head(14).style.set_table_styles([{
        'selector':'th', 'props':[('background-color','#c00000'),('color','white')]
    }])


# In[42]:


### Guardar Achivos
Data_Habil.to_csv('RESUMEN_CENTROS_HORA_'+str(Y*10000+M*100+D)+'_HABIL.csv',sep=';', index=False)
Data_Habil_Res.to_csv('RESUMEN_CENTROS_'+str(Y*10000+M*100+D)+'_HABIL.csv',sep=';', index=False)


# ### FDS

# In[43]:


Data_fds= DataTotal[(DataTotal['BASE']=='principal')&(DataTotal['FILTRO']=='fds')&(DataTotal['FECHA']>=minFechaFDS)&(DataTotal['FECHA']<maxFecha)].groupby(['ID','HORA']).agg(
                CONTEO=('ID','count'),
                SUBIDA=('VELOCIDAD_SUBIDA', quantiles_adaptados),
                BAJADA=('VELOCIDAD_BAJADA', quantiles_adaptados)
            )


# In[44]:


Data_fds.reset_index(inplace=True)
BANDA=Data[['ID','BANDA','SUBIDA_LIM','BAJADA_LIM']].drop_duplicates(subset='ID',keep='last')
Data_fds=Data_fds.merge(BANDA,left_on='ID',right_on='ID', how='left')
Data_fds.rename(columns={'BAJADA_LIM':'BAJA','SUBIDA_LIM':'SUBE','BAJADA':'BAJA_R','SUBIDA':'SUBE_R'}, inplace=True)
Data_fds=Data_fds[['ID','BANDA','HORA','BAJA','SUBE','BAJA_R','SUBE_R']]


# In[45]:


Data_fds['CUMPLE_BAJA']=np.where(Data_fds['BAJA_R']>=Data_fds['BAJA'],'SI','NO')
Data_fds['CUMPLE_SUBE']=np.where(Data_fds['SUBE_R']>=Data_fds['SUBE'],'SI','NO')
Data_fds['CUMPLE']=np.where( (Data_fds['CUMPLE_BAJA']=='SI') & (Data_fds['CUMPLE_SUBE']=='SI'),'SI','NO')
Data_fds['PORC_BAJADA']=Data_fds['BAJA_R']/Data_fds['BAJA']
Data_fds['PORC_SUBE']=Data_fds['SUBE_R']/Data_fds['SUBE']
Data_fds['CERCANO_BAJA']=np.where(Data_fds['PORC_BAJADA']>1, 'CUMPLE', np.where(Data_fds['PORC_BAJADA']>0.9,'SI','NO'))
Data_fds['CERCANO_SUBE']=np.where(Data_fds['PORC_SUBE']>1, 'CUMPLE', np.where(Data_fds['PORC_SUBE']>0.9,'SI','NO'))
Data_fds['CERCANO']=np.where( (Data_fds['CERCANO_BAJA']=='CUMPLE') & (Data_fds['CERCANO_SUBE']=='CUMPLE'),'CUMPLE',
                                 np.where((Data_fds['CERCANO_BAJA']=='SI') & (Data_fds['CERCANO_SUBE']=='SI'),'AMBOS',
                                         np.where((Data_fds['CERCANO_BAJA']=='CUMPLE') & (Data_fds['CERCANO_SUBE']=='SI'),'SUBE',
                                                 np.where((Data_fds['CERCANO_BAJA']=='SI') & (Data_fds['CERCANO_SUBE']=='CUMPLE'),'BAJA','NO'))))


# In[46]:


Data_fds_Res=Data_fds.copy()
Data_fds_Res['TIPO2']=np.where(Data_fds_Res['CERCANO']=="NO",'NO','SI')
Data_fds_Res=pd.pivot_table(Data_fds_Res[['ID','BANDA','CUMPLE']], index=['ID','BANDA'], columns=['CUMPLE'], aggfunc=len, fill_value=0)
Data_fds_Res['PORC']=Data_fds_Res['NO']/(Data_fds_Res['NO']+Data_fds_Res['SI'])
Data_fds_Res['TIPO']= np.where(Data_fds_Res['PORC']==1,'CRITICO',np.where(Data_fds_Res['PORC']==0,'BUENO','REGULAR'))


# In[47]:


Data_fds.to_csv('RESUMEN_CENTROS_HORA_'+str(Y*10000+M*100+D)+'_FDS.csv',sep=';',index=False)
Data_fds_Res.to_csv('RESUMEN_CENTROS_'+str(Y*10000+M*100+D)+'_FDS.csv',sep=';',index=False)


# # Maestro Aps

# In[48]:


maestro_Aps=Data[['ID','BANDA','BEAM','FASE']].drop_duplicates()


# In[49]:


maestro_Aps['BEAM'].value_counts()


# ## Funcion Asignacion

# In[50]:


def asignarMuestras(Data,num_muestras_hora):
    DataExito=Data[Data['CONDICION']=='E']    
    #preprocesamiento
    DataExito['IND']=((DataExito['VELOCIDAD_SUBIDA']-DataExito['SUBIDA_LIM'])+(DataExito['VELOCIDAD_BAJADA']-DataExito['BAJADA_LIM']))/2
    
    tabla_ind=DataExito.groupby(['ID','HORA'])[['IND']].median()
    tabla_ind.reset_index(inplace=True)
    tabla_freq=DataExito.groupby(['ID','HORA'])[['CONDICION']].count().reset_index()

    #calculo cumplimiento para mejores y peores casos
    tabla_ind_cump=pd.pivot_table(DataExito[['ID','HORA','CUMPLE_BANDA','CONDICION']], index=['ID','HORA'], columns=['CUMPLE_BANDA'], aggfunc=len, fill_value=0)
    tabla_ind_cump.columns=tabla_ind_cump.columns.droplevel()
    tabla_ind_cump.reset_index(inplace=True)
    tabla_ind_cump['ind_cump']=tabla_ind_cump['SI']/(tabla_ind_cump['SI']+tabla_ind_cump['NO'])
    
    tablaBase=tabla_ind.merge(tabla_freq, left_on=['ID','HORA'], right_on=['ID','HORA'], how='left')
    tablaBase=tablaBase.merge(tabla_ind_cump[['ID','HORA','ind_cump']], left_on=['ID','HORA'], right_on=['ID','HORA'], how='left')
    
    tablaBase.columns=['ID','HORA','IND','FREQ','ind_cump']
    tablaBase.reset_index(inplace=True)
    
    tablaBase_ori=tablaBase.copy()
    
    #Distribución aps bajos
    tablaBase.sort_values(['ID','IND','FREQ'], ascending=False, inplace=True)
    tablaBase.reset_index(drop=True, inplace=True)
    tabla_primera_opcion=tablaBase.groupby(['ID']).nth(0)
    casos_bajos=tabla_primera_opcion[tabla_primera_opcion['IND']<0]
    muestra_bajas_hora=math.ceil(len(casos_bajos)/num_horas)

    casos_bajos=casos_bajos.sort_values('IND')
    casos_bajos.reset_index(inplace=True)
    casos_bajos.reset_index(inplace=True)
    #casos_bajos
    
    #lograr la mayor distribución de hora posible
    casos_bajos['franja']=np.where(casos_bajos['level_0']//num_horas%2==0,casos_bajos['level_0']%num_horas,(num_horas-1-casos_bajos['level_0']%num_horas))
    casos_bajos['hora']=casos_bajos['franja']+6
    tablaBase=tablaBase[~tablaBase['ID'].isin(casos_bajos['ID'].unique())]

    num_muestras_hora=num_muestras_hora-muestra_bajas_hora
    asignacionFinal=pd.DataFrame()

    #Asignacion Principal
    i=0
    num_horas_sobre=1
    while num_horas_sobre > 0:   
        #Reordenar la base
        tablaBase.sort_values(['ID','IND','FREQ'], ascending=False, inplace=True)
        tablaBase.reset_index(drop=True, inplace=True)
        tablaBase['index']=tablaBase.index

        tablaBase['dif']=tablaBase.groupby('ID')['IND'].diff(-1)
        tablaBase['index_N']=tablaBase.groupby('ID')['index'].shift(periods=-1)
        tablaBase['HORA_N']=tablaBase.groupby('ID')['HORA'].shift(periods=-1)
        tablaBase['IND_N']=tablaBase.groupby('ID')['IND'].shift(periods=-1)
        tablaBase['FREQ_N']=tablaBase.groupby('ID')['FREQ'].shift(periods=-1)

        tabla_primera_opcion=tablaBase.groupby(['ID']).nth(0)
        tabla_primera_opcion.reset_index(inplace=True)

        tabla_ronda=tabla_primera_opcion

        resumen=tabla_ronda[['HORA']].value_counts().to_frame()
        resumen['DIF']=resumen[0]-num_muestras_hora

        horas_sobre=resumen[resumen['DIF']>0]           
        num_horas_sobre=len(horas_sobre)

        # guardar valores
        if num_horas_sobre>0:
            hora_aux=horas_sobre.index[0][0]
            dif_aux=horas_sobre.iloc[0]['DIF']
            tabla_ronda=tabla_ronda.sort_values('dif')
            saldo=tabla_ronda[tabla_ronda['HORA']==hora_aux][dif_aux:]        
            if i==0:
                asignacionFinal=saldo
            else: 
                asignacionFinal=asignacionFinal.append(saldo)            

            tablaBase=tablaBase[tablaBase['HORA']!=hora_aux]
            tablaBase=tablaBase[~tablaBase['ID'].isin(saldo['ID'])]            
            i+=1

    if i==0:
        asignacionFinal=tabla_ronda
    else: 
        asignacionFinal=asignacionFinal.append(tabla_ronda)


    # Matriz de posiciciones a llenar
    num_muestras_hora=num_muestras_hora+muestra_bajas_hora
    tabla_final=pd.DataFrame({'hora':list(range(6,20+1))},columns=['hora']+list(range(num_muestras_hora)))
    tabla_final.set_index('hora',inplace=True)
    tabla_final['pos']=0

    #Matriz de causa asignacion para asignación de satelitales
    tabla_final_causa=pd.DataFrame({'hora':list(range(6,20+1))},columns=['hora']+list(range(num_muestras_hora)))
    tabla_final_causa.set_index('hora',inplace=True)
    
    # Asignar resultado anterior
    for index, row in asignacionFinal.iterrows():    
        hora_aux=int(row['HORA'])
        pos=tabla_final.at[hora_aux,'pos']    
        tabla_final.at[hora_aux,pos]=int(row['ID'])    
        tabla_final.at[hora_aux,'pos']=tabla_final.at[hora_aux,'pos']+1
        tabla_final_causa.at[hora_aux,pos]='1.m_aps'
    
    #Asignar casos bajos
    for ind,row in casos_bajos.iterrows():
        t_aux=tabla_final.copy()
        t_aux=t_aux[(t_aux['pos']<num_muestras_hora)]
        sel_hora=row['hora']
        idx=int(row['ID'])
        pos=tabla_final.at[sel_hora,'pos']
        tabla_final.at[sel_hora,pos]=idx    
        tabla_final.at[sel_hora,'pos']=tabla_final.at[sel_hora,'pos']+1
        tabla_final_causa.at[sel_hora,pos]='2.bajos'
    
    #Validar aps unicos
    ##print(tabla_final.reset_index().drop(columns=['pos']).melt(id_vars=['hora'],var_name='muestra',value_name='ID').nunique())
    #Agregar 100% F
    id_asignados=list(casos_bajos['ID'].unique())+list(asignacionFinal['ID'].unique())
    APs_sin_Exito=Data[~Data['ID'].isin(id_asignados)]
    APs_sin_Exito=list(APs_sin_Exito['ID'].unique())
    
    for idx in APs_sin_Exito:
        t_aux=tabla_final.copy()
        t_aux=t_aux[(t_aux['pos']<num_muestras_hora)]
        t_aux=t_aux.sort_values('pos',ascending=True)
        sel_hora=t_aux.head(1).index[0]
        pos=tabla_final.at[sel_hora,'pos']
        tabla_final.at[sel_hora,pos]=idx    
        tabla_final.at[sel_hora,'pos']=tabla_final.at[sel_hora,'pos']+1
        tabla_final_causa.at[sel_hora,pos]='3.sin_E'
    
    #Asignar mejores casos
    tablaBase=tabla_ind.merge(tabla_freq, left_on=['ID','HORA'], right_on=['ID','HORA'], how='left')    
    tablaBase=tablaBase.merge(tabla_ind_cump[['ID','HORA','ind_cump']], left_on=['ID','HORA'], right_on=['ID','HORA'], how='left')
    tablaBase.columns=['ID','HORA','IND','FREQ','ind_cump']
    tablaBase.reset_index(inplace=True)    
    mejores_hora=tablaBase[(tablaBase['ind_cump']==1)&(tablaBase['FREQ']>=3)]
    group_casos_mejora=mejores_hora.groupby('ID')

    i=0
    for g in group_casos_mejora.groups:
        i=i+1
        group = group_casos_mejora.get_group(g)
        val_ids=group['HORA'].values
        t_aux=tabla_final.copy()
        t_aux=t_aux[(t_aux['pos']<num_muestras_hora)&(t_aux.index.isin(val_ids))]
        t_aux= t_aux[~t_aux.isin([g]).any(axis=1)]

        for sel_hora in t_aux.index:
            pos=tabla_final.at[sel_hora,'pos']
            tabla_final.at[sel_hora,pos]=int(g)  
            tabla_final.at[sel_hora,'pos']=tabla_final.at[sel_hora,'pos']+1
            tabla_final_causa.at[sel_hora,pos]='4.mc_hora'
    
    #Completar faltantes con mejores casos
    tabla_ind_total=pd.pivot_table(DataExito[['ID','CUMPLE_BANDA','CONDICION']], index=['ID'], columns=['CUMPLE_BANDA'], aggfunc=len, fill_value=0)
    tabla_ind_total.columns=tabla_ind_total.columns.droplevel()
    tabla_ind_total.reset_index(inplace=True)
    tabla_ind_total['ind']=tabla_ind_total['SI']/(tabla_ind_total['SI']+tabla_ind_total['NO'])
    tabla_ind_total=tabla_ind_total.sort_values(['ind','SI'], ascending=False)
    pos_mejores=0
    resultado_mejores_casos=tabla_ind_total[tabla_ind_total['ind']==1].merge(maestro_Aps, left_on='ID', right_on='ID', how='left')

    while len(tabla_final[tabla_final['pos']<num_muestras_hora])>0:
        t_aux=tabla_final[tabla_final['pos']<num_muestras_hora]
        id_aux=tabla_ind_total['ID'].values[pos_mejores]
        t_aux= t_aux[~t_aux.isin([id_aux]).any(axis=1)]

        for sel_hora in t_aux.index:
            pos=tabla_final.at[sel_hora,'pos']
            tabla_final.at[sel_hora,pos]=int(id_aux)  
            tabla_final.at[sel_hora,'pos']=tabla_final.at[sel_hora,'pos']+1
            tabla_final_causa.at[sel_hora,pos]='5.mc'
        pos_mejores=pos_mejores+1

    ## validar duplicados por hora
    for hora,row in tabla_final.iterrows():    
        a_set = set(row)
        contains_duplicates = len(row) != len(a_set)
        print(hora,contains_duplicates)
    
    #Alistamiento de salida
    tabla_ind_total['tipo']=np.where(tabla_ind_total['ind']==0, 'CRITICO', np.where(tabla_ind_total['ind']==1,'BUENO','REGULAR'))
    tabla_final_un=tabla_final.reset_index().drop(columns=['pos']).melt(id_vars=['hora'],var_name='muestra',value_name='ID')
    tabla_final_un=tabla_final_un.merge(tabla_ind_total, left_on='ID', right_on='ID', how='left')
    tabla_final_un['tipo']=np.where(tabla_final_un['tipo'].isnull(),'CRITICO',tabla_final_un['tipo'])
    tabla_final_un=tabla_final_un.merge(maestro_Aps, left_on='ID', right_on='ID', how='left')
    # Agregar indicador por horas
    tabla_final_un=tabla_final_un.merge(tablaBase_ori, left_on=['ID','hora'], right_on=['ID','HORA'], how='left')
    
    #agregar causas
    tabla_final_un_causas=tabla_final_causa.reset_index().melt(id_vars=['hora'],var_name='muestra',value_name='causa')
    tabla_final_un=tabla_final_un.merge(tabla_final_un_causas, left_on=['hora','muestra'], right_on=['hora','muestra'], how='left')
    #fin de la funcion,  retorno
    return tabla_final_un,resultado_mejores_casos


# In[51]:


def validarSatelitales(tablaBase,tablaMejores):
    tabla_fina_com=tablaBase.copy()
    mc_fina_com=tablaMejores.copy()
    tabla_fina_com.reset_index(drop=True,inplace=True)
    #Resultado número de muestras por horas de satelitales  (beams diferente a -1)
    df_muestras=tabla_fina_com[tabla_fina_com['BEAM']!=-1][['hora','BEAM']].value_counts().to_frame().reset_index()
    df_muestras=df_muestras[df_muestras[0]>n_beams_sat]    
    mc_fina_com=mc_fina_com[mc_fina_com['BEAM']==-1]
    
    res_mc=mc_fina_com[['BANDA','FASE']].value_counts().to_frame().reset_index()
    # Iteración
    for key,row in df_muestras.iterrows():
        (aux_hora,aux_beam,aux_num_casos)=row.values
        ind_rev=tabla_fina_com[tabla_fina_com['hora']==aux_hora]['ID'].values
        df_banda_hora_sobre=tabla_fina_com[(tabla_fina_com['hora']==aux_hora)&(tabla_fina_com['BEAM']==aux_beam)].sort_values('causa', ascending=False)
        exceso=aux_num_casos-n_beams_sat
        df_exceso=df_banda_hora_sobre[(df_banda_hora_sobre['causa']=='4.mc_hora')|(df_banda_hora_sobre['causa']=='5.mc')][:exceso]
        res_exceso=df_exceso[['BANDA','FASE']].value_counts().to_frame().reset_index()
        reemplazos=pd.DataFrame(columns=mc_fina_com.columns)
        for key,row in res_exceso.iterrows():    
            df_bolsa_reemplazos=mc_fina_com[(mc_fina_com['BANDA']==row['BANDA']) & (mc_fina_com['FASE']==row['FASE'])]
            df_bolsa_reemplazos=df_bolsa_reemplazos[~df_bolsa_reemplazos['ID'].isin(ind_rev)]    
            res_mc_rem=df_bolsa_reemplazos[['BANDA','FASE']].value_counts().to_frame().reset_index()    
            num_bolsa=len(df_bolsa_reemplazos)
            reemplazos=reemplazos.append(df_bolsa_reemplazos.sample(min(row[0],num_bolsa)))
        reemplazos['hora']=aux_hora
        reemplazos['causa']='5.mc'
        tabla_fina_com=tabla_fina_com[~tabla_fina_com.index.isin(df_exceso.index)]
        tabla_fina_com=tabla_fina_com.append(reemplazos)
        tabla_fina_com.reset_index(drop=True, inplace=True)
    return tabla_fina_com


# ###  Distribucion de las muestras en las bandas

# In[52]:


distribucion=Data[['ID','BANDA','FASE']].drop_duplicates()
res_dist=distribucion[['BANDA','FASE']].value_counts().to_frame()
res_dist['tamMuestra']=res_dist[0]*pc_muestra
res_dist['tamMuestra_int']=res_dist['tamMuestra'].apply(np.ceil).astype(int)
res_dist['tamMuestra_int'].sum()
res_dist.head(1).index[0]


# In[53]:


(num_muestras_hora-res_dist['tamMuestra_int'].sum())


# In[54]:


num_muestras_hora


# In[55]:


## Ajustar tamaño
res_dist.at[res_dist.head(1).index,'tamMuestra_int']=res_dist.at[res_dist.head(1).index[0],'tamMuestra_int']+(num_muestras_hora-res_dist['tamMuestra_int'].sum())


# In[56]:


res_dist['tamMuestra_int'].sum()


# In[57]:


res_dist.reset_index(inplace=True)


# In[58]:


res_dist['minimo']=res_dist[0]/num_horas


# In[59]:


res_dist


# ### Habiles

# In[60]:


tabla_fina_com=pd.DataFrame()
mc_fina_com=pd.DataFrame()
i=0
for key,row in res_dist.iterrows():
    print(row['BANDA'],row['FASE'])
    (tabla_final_un,resultado_mejores_casos)=asignarMuestras(Data[(Data['FILTRO']=='hab')&(Data['BANDA']==row['BANDA'])&(Data['FASE']==row['FASE'])],row['tamMuestra_int'])    
    if i==0:
        tabla_fina_com=tabla_final_un.copy()
        mc_fina_com=resultado_mejores_casos.copy()
    else :
        tabla_fina_com=tabla_fina_com.append(tabla_final_un)
        mc_fina_com=mc_fina_com.append(resultado_mejores_casos)        
    i=i+1


# In[61]:


sum(tabla_fina_com[['hora','ID']].value_counts()>1)


# In[62]:


tabla_fina_com['ID'].nunique()


# In[63]:


pd.crosstab(index=tabla_fina_com['hora'], columns=tabla_fina_com['BEAM'], margins=True)


# In[64]:


tabla_fina_com.groupby('BEAM')['ID'].nunique()


# In[65]:


tabla_fina_com=validarSatelitales(tabla_fina_com,mc_fina_com)


# In[66]:


sum(tabla_fina_com[['hora','ID']].value_counts()>1)


# In[67]:


tabla_fina_com.groupby('BEAM')['ID'].nunique()


# In[68]:


pd.crosstab(index=tabla_fina_com['hora'],
            columns=tabla_fina_com['tipo'], margins=True)


# In[69]:


pd.crosstab(index=tabla_fina_com['hora'], columns=tabla_fina_com['BEAM'], margins=True)


# In[70]:


tabla_fina_com['causa'].value_counts()


# In[71]:


tabla_fina_com['ID'].nunique()


# In[72]:


pd.crosstab(index=tabla_fina_com['hora'], columns=tabla_fina_com['FASE'], margins=True)


# In[73]:


archivo=tabla_fina_com[['ID','BANDA','hora']]
archivo.columns=['ID','BANDA','HORA']
archivo=archivo.sort_values('HORA')


# In[74]:


archivo.isnull().sum()


# In[75]:


duplicateRowsDF = archivo[archivo.duplicated()]
duplicateRowsDF


# In[76]:


archivo_hab=archivo.copy()


# In[77]:


final_hab=archivo.copy()


# In[78]:


final_hab['ID'].nunique()


# In[79]:


final_hab.groupby('BANDA')['ID'].nunique()


# In[80]:


res_dist['tamMuestra_int'].sum()


#  ### Festivos

# In[81]:


tabla_fina_com=pd.DataFrame()
mc_fina_com=pd.DataFrame()
i=0
for key,row in res_dist.iterrows():
    print(row['BANDA'],row['FASE'])
    (tabla_final_un,resultado_mejores_casos)=asignarMuestras(Data[(Data['FILTRO']=='fds')&(Data['BANDA']==row['BANDA'])&(Data['FASE']==row['FASE'])],row['tamMuestra_int'])
    if i==0:
        tabla_fina_com=tabla_final_un.copy()
        mc_fina_com=resultado_mejores_casos.copy()
    else :
        tabla_fina_com=tabla_fina_com.append(tabla_final_un)
        mc_fina_com=mc_fina_com.append(resultado_mejores_casos) 
    i=i+1


# In[82]:


tabla_fina_com=validarSatelitales(tabla_fina_com,mc_fina_com)


# In[83]:


pd.crosstab(index=tabla_fina_com['hora'],
            columns=tabla_fina_com['tipo'], margins=True)


# In[84]:


pd.crosstab(index=tabla_fina_com['hora'], columns=tabla_fina_com['BEAM'], margins=True)


# In[85]:


pd.crosstab(index=tabla_fina_com['hora'], columns=tabla_fina_com['FASE'], margins=True)


# In[86]:


sum(tabla_fina_com[['hora','ID']].value_counts()>1)


# In[87]:


tabla_fina_com['causa'].value_counts()


# In[88]:


tabla_fina_com['ID'].nunique()


# In[89]:


archivo=tabla_fina_com[['ID','BANDA','hora']]
archivo.columns=['ID','BANDA','HORA']
archivo=archivo.sort_values('HORA')


# In[90]:


archivo.isnull().sum()


# In[91]:


duplicateRowsDF = archivo[archivo.duplicated()]
duplicateRowsDF


# In[92]:


archivo_fds=archivo.copy()


# In[93]:


final_fds=archivo.copy()


# In[94]:


final_fds['ID'].nunique()


# In[95]:


final_hab.groupby('BANDA')['ID'].nunique()


# ## Guardar Archivos

# In[96]:


archivo_hab.to_csv(PathHab,sep=';', index=False)


# In[97]:


archivo_fds.to_csv(PathFds,sep=';', index=False)


# # Prueba Aleatoria
# * Realizar asignacion aleatoria
# * validar indicadores principales
# * calcular número de franjas cumplidas en los 2 escenarios

# In[98]:


APs_id=DataTotal['ID'].unique()
len(APs_id)


# In[99]:


tabla_final=pd.DataFrame({'hora':list(range(6,20+1))},columns=['hora']+list(range(num_muestras_hora)))
tabla_final.set_index('hora',inplace=True)
tabla_final['pos']=0


# ### Union con muestreos

# In[100]:


DataTotal_validacion=DataTotal.copy()


# In[101]:


DataTotal_validacion=DataTotal_validacion.merge(final_fds.rename(columns={'ID':'ID_fds'}), left_on=['HORA','ID'], right_on=['HORA','ID_fds'],how='left')


# In[102]:


DataTotal_validacion=DataTotal_validacion.merge(final_hab.rename(columns={'ID':'ID_hab'}), left_on=['HORA','ID'], right_on=['HORA','ID_hab'],how='left')


# In[103]:


DataTotal_validacion[DataTotal_validacion['ID_hab'].notnull()]


# In[104]:


DataTotal_validacion['ID_analitica']=np.where(DataTotal_validacion['FILTRO']=='hab',DataTotal_validacion['ID_hab'],DataTotal_validacion['ID_fds'])


# ### Resumen indicadores
# *  No se paga por esta medida

# In[105]:


Resultados_analitica=DataTotal_validacion[(DataTotal_validacion['ID_analitica'].notnull())]


# In[106]:


Resultados_analitica['ID_analitica'].nunique()


# #### EXITO / FRACASO

# In[107]:


if con_graficas:
    pieChart(Resultados_analitica['CONDICION'])


# #### CUMPLE BANDA

# In[108]:


if con_graficas:
    pieChart(Resultados_analitica[Resultados_analitica['CONDICION']=='E']['CUMPLE_BANDA'])


# ### Subida

# In[109]:


Resultados_analitica[Resultados_analitica['CONDICION']=='E']['CUMPLE_SUBIDA'].value_counts()/len(Resultados_analitica[Resultados_analitica['CONDICION']=='E'])


# In[110]:


if con_graficas:
    pieChart(Resultados_analitica[Resultados_analitica['CONDICION']=='E']['CUMPLE_SUBIDA'])


# ## Bajada

# In[111]:


Resultados_analitica[Resultados_analitica['CONDICION']=='E']['CUMPLE_BAJADA'].value_counts()/len(Resultados_analitica[Resultados_analitica['CONDICION']=='E'])


# In[112]:


if con_graficas:
    pieChart(Resultados_analitica[Resultados_analitica['CONDICION']=='E']['CUMPLE_BAJADA'])


# In[113]:


Resultados_analitica['BASE'].value_counts()/len(Resultados_analitica)


# ## Cumplimiento Franjas

# In[114]:


dic_cumplimiento={'DL:12.00Mb-UL:3.00Mb':[12000,3000],'DL:15.00Mb-UL:3.75Mb':[15000,3750],'DL:18.00Mb-UL:4.50Mb':[18000,4500],'DL:21.00Mb-UL:5.25Mb':[21000,5250]}
dic_cumplimiento=pd.DataFrame.from_dict(dic_cumplimiento, orient='index')
dic_cumplimiento.columns=['Lim_bajada','Lim_subida']
dic_cumplimiento


# In[115]:


res_franjas_analitica=Resultados_analitica[Resultados_analitica['CONDICION']=='E'].groupby(['BANDA_x','HORA'])['VELOCIDAD_BAJADA','VELOCIDAD_SUBIDA'].quantile(0.05).reset_index()
res_franjas_analitica=res_franjas_analitica.merge(dic_cumplimiento,left_on='BANDA_x', right_index=True, how='left')
res_franjas_analitica['cump_bajada']=np.where(res_franjas_analitica['VELOCIDAD_BAJADA']>=res_franjas_analitica['Lim_bajada'],1,0)
res_franjas_analitica['cump_subida']=np.where(res_franjas_analitica['VELOCIDAD_SUBIDA']>=res_franjas_analitica['Lim_subida'],1,0)
res_franjas_analitica


# In[116]:


res_franjas_analitica.groupby('BANDA_x')['cump_bajada','cump_subida'].sum()







### Agregado en implementación

ssh = SSH()
ssh.stpf(PathHab,PathHab)
ssh.stpf(PathFds,PathFds)

dateModFileHab=time.strftime('%Y/%m/%d', time.gmtime(os.path.getmtime(PathHab)))
dateModFileFds=time.strftime('%Y/%m/%d', time.gmtime(os.path.getmtime(PathFds)))

if dayNotify in dayHabNotify and (todayNotify not in ResultNotify):
    if todayNotify != dateModFileHab:
        message = 'No se actualizó el archivo de entrada desde SQL: '+PathHab
        subject = 'Error Script Analítica'
        notify.WsTelegram(message)
        notify.WsEmail(subject,message)
        time.sleep(30)
        
elif (dayNotify in dayFdsNotify) or (todayNotify in ResultNotify):
    if todayNotify != dateModFileFds:
        message = 'No se actualizó el archivo de entrada desde SQL: '+PathFds
        subject = 'Error Script Analítica'
        notify.WsTelegram(message)
        notify.WsEmail(subject,message)
        time.sleep(30)



