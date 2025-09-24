#!/usr/bin/env python
# coding: utf-8

# In[1]:

###############################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
###############################################################################

# # Tutorial for pySTAR 
# Author: Dimitrios Fardis  
# Maintainer: Dimitrios Fardis  
# Updated: 10/18/2024
# 
# **Python Symbolic regression Through Algebraic Representations** (pySTAR) provides tools for generating mathematical (symbolic) models that best fit some input data (Sarwar, 2022). The pySTAR toolbox allows for the definition of the form of the surrogate model and the regression parameter values simultaneously.
# 
# ## References
# Sarwar, O. 2022, Algorithms for Interpretable High-Dimensional Regression, Carnegie Mellon University.

# ## 1. Installation
# 
# The **pySTAR** toolbox is installed by default as part of IDAES. For instructions on installing IDAES, see the [online documentation](https://idaes-pse.readthedocs.io/en/stable/).

# ## 2. Generating surrogate models with pySTAR
# 
# The pySTAR framework currently provides tools for generating surrogates that allow for seven types of operators in the mathematical expression. Four of the operators are binary operators:
# 
# - Addition (+)
# - Substraction (-)
# - Multiplication (*)
# - Division (/)
# 
# The rest three operators are unary:
# - Square root (^0.5)
# - Logarithm (log)
# - Exponential (exp)

# ### 3. Generating symbolic regression models for the outputs of a leaching process
# 
# As an example, let us generate a surrogate model for an output of the leaching process of a critical minerals recovery plant. During this process, various critical minerals are extracted from ores using sulfuric acid. Two streams enter the leaching process of the plant; a solids stream that contain the ores and a liquid stream with an aqueous solution of sulfuric acid. The outlets of leaching are two; a solids waste stream and a liquid stream that is enriched with the critical minerals in ionic form.
# 
# The input variables of this process are two; the molar concentration of sulfuric acid in the liquid inlet and the mass flowrate of the solids stream. The output variables of the process are thirty. All the variables, as well as ten datapoints, can be found in the csv file '3_4_simulation_data_{k}.csv'. The combined dataset (training and testing points) have been min-max scaled between the numbers 3 and 4. The pySTAR algorithm performs better when the data are scaled in this way. Scaling between 0 and 1 would cause problems with the division operation.
# 
# As an example, let us generate a surrogate model for the volumetric flowrate of the liquid outlet which is an output of the leaching process, named 'liquid_outlet_flow_vol'. 

# ### 3.1. Training the model

# #### Step 1: Import the needed packages and the symbolic regression tool

# In[2]:


import numpy as np
import time
from pyomo.opt import SolverFactory
import os
import csv
import hashlib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from STEP_1 import *
from STEP_2 import *
from STEP_3 import *

# #### Step 2: Read the txt file with the name of the output
# 
# The txt file 'output.txt' contains the name of the output variable. In this example, the name of the output is 'liquid_outlet_flow_vol'.

# In[3]:

############Output name##############
# Open the txt file in read mode
with open('output.txt', 'r') as file:
    # Read the contents of the file
    content = file.read().strip()
# Convert the contents to an integer and assign to model
output_name = content
# Print the model type to verify
print('Output=',output_name)

########Training points###########
# Open the txt file in read mode
with open('input.txt', 'r') as file:
    # Read the contents of the file
    content = file.read().strip()
# Convert the contents to an integer and assign to k
k = int(content)
# Print the value of k to verify
print('Training points=',k)

# #### Step 3: Read the csv file with our data using pandas
# We also print the 10 points of our dataset. The variables 'liquid_inlet_conc_mol_comp_H2SO4' and 'solid_inlet_flow_mass' are the inputs of the leaching process and the variable 'liquid_outlet_flow_vol' is an output.

# In[4]:

# read in our csv data
columns = ['liquid_inlet_conc_mol_comp_H2SO4','solid_inlet_flow_mass',
           f'{output_name}'] 
         
df = pd.read_csv(f'3_4_simulation_data_{k}.csv', usecols=columns)

# separate the data into inputs and outputs
inputs = ['liquid_inlet_conc_mol_comp_H2SO4','solid_inlet_flow_mass']

outputs= [f'{output_name}']

dfin = df[inputs]
dfout = df[outputs]
print('Inputs:')
print(dfin)
print('Output:')
print(dfout)

#Convert the pandas dataframes to numpy arrays
X = dfin.to_numpy()
y = dfout.to_numpy()

# #### Step 4: Solve NLP (MINLP relaxed symbolic regression tree problem)
# Then, the options of the BARON solver are selected for solving the MINLP relaxed symbolic regression tree problem. After solving the NLP problem, a csv file is created named 'Yr_{dataset_name}.csv' in the folder 'Results_Yr'. The csv file contains the probabilities Yr of assigning an operator or operand to a specific node of the tree. 
# 
# The user has to specify a name for the variable 'dataset_name'.
# 
# The user can change BARON's option, solver.options['PrLevel'] to equal '1', in order to print the log output.

# In[5]:


np.random.seed(42)
cutoff=10000
np.seterr(divide='ignore', invalid='ignore')

#BARON as the solver and its options
solver = SolverFactory('baron')
solver.options['DeltaTerm'] = 1 # 0 is the default
solver.options['DeltaT'] = 60
solver.options['DeltaR'] = 0.01
solver.options['AllowIPOPT'] = 0 # 1 is the default
solver.options['PrLevel'] = 1   #Set as 1 (default) to print the log output

dataset_name = f'{k}_{output_name}'
print('Dataset name:', dataset_name)
print()

# Define the folder path
folder_path_cosntant = "Results_constant_fit_trees"
# Create the folder if it doesn't exist
if not os.path.exists(folder_path_cosntant):
    os.makedirs(folder_path_cosntant)

#Check if the same thing was done before and avoid if it was
file_path_constant_fit_trees = f"Results_constant_fit_trees/constant_fit_trees_{dataset_name}.csv"

#scaler = MinMaxScaler((3,4))
#X = scaler.fit_transform(X)
#y = scaler.fit_transform(y.reshape(-1, 1))

#print(X)
#for element in y:
    #print(element)

#STEP 1 -------------------------------------------------------------
#Significant part of the implementation of STEP 1 has been written by Owais Sarwar and Mina Kim
#Solve NLP (relaxed symbolic regression tree problem)
starting_time = time.perf_counter()
mr, Yr, Br, Ur, Lr, NnotTr, Tr, Nr, c_lo, c_up, depth_symbol = SRT(X,y)

solver.options['MaxTime'] = -1
print('Relaxed MINLP:')
#solver.options['summary'] = 1 
#solver.options['SumName'] = f'summary_relaxed_MINLP'
results = solver.solve(mr, tee=True, symbolic_solver_labels=True, keepfiles=True)
depth = value(depth_symbol)
print('Running BARON to optimize to solve MINLP relaxation')
print('Solver Status:', results.solver.status)
print('Terimation Condition:', results.solver.termination_condition)

# The solver output file path
#solver_output_file = f'summary_relaxed_MINLP'
#disp(FileLink(solver_output_file))

MINLP_end_time = time.perf_counter()
MINLP_total_time = MINLP_end_time - starting_time
print('CPU total time for relaxed MINLP=', MINLP_total_time)

# Define the folder path
folder_path = "Results_Yr"
# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Write the results to a csv file
file_path_Yr = f"Results_Yr/Yr_{dataset_name}.csv" 
with open(file_path_Yr, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index in sorted(Yr, key=lambda x: (x[0], x[1])):
        writer.writerow([index, value(mr.y[index])])


# #### Step 5: Generate a group of candidate expression trees using the probabilities Yr
# The user has to specify the number of trees created. In this example, we create 100 expression trees. The 100 tree expressions are saved in a csv file named 'trees_{dataset_name}.csv' in the folder 'Results_trees'.

# In[6]:


# # #STEP 2 -------------------------------------------------------------
#Significant part of the implementation of STEP 2 has been written by Owais Sarwar and Mina Kim
#Specify the number of trees created
num_trees = 100

#Generate a group of trees using the result of solving NLP as probability
trees = candidate_trees(X, mr, Yr, Br, Ur, Lr, NnotTr, Tr, Nr, num_trees)

# Define the folder path
folder_path_trees = "Results_trees"
# Create the folder if it doesn't exist
if not os.path.exists(folder_path_trees):
    os.makedirs(folder_path_trees)

#Write the results to a csv file
file_path_trees = f"Results_trees/trees_{dataset_name}.csv" 
with open(file_path_trees, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for tree in trees:
        writer.writerow(tree)


# #### Step 6: Fit the constant values of each tree using regression
# 
# The user can change BARON's option, solver.options['PrLevel'] to equal '1', in order to print the log output for the optimization of each tree.
# 
# For each BARON's run, summary files that contain information for the optimization problem are created. The user can press on the links to see the contents of the files that are saved on his local computer.

# In[7]:


#STEP 3 -------------------------------------------------------------
#Significant part of the implementation of STEP 3 has been written by Owais Sarwar and Mina Kim
#Read the csv files written from STEP 2 and make the trees unique
unique_trees = {}
with open(file_path_trees, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Yfix, es, cst_count = row
        cst_count = int(cst_count)
        es_hash = hashlib.sha256(es.encode()).hexdigest()  ##Hashing, could be changed with CRC function
        if es_hash not in unique_trees:
            unique_trees[es_hash] = (Yfix, es, cst_count)
    unique_trees = list(unique_trees.values())

#Fit the constant values of each candidate tree and find the best model with smallest RMSE
min_rmse = float('inf') 
best_model_dataset_name = None
results = []

solver.options['MaxIter'] = 10  
solver.options['PrLevel'] = 1   #Set as 1 (default) to print the log output
solver.options['FirstLoc'] = 1         ################# Stop at first local optimum ##################
CPU_total_start_time = time.perf_counter()
for tree, (Yfix, es, cst_count) in enumerate(unique_trees):
    print('Tree:', tree)
    #solver.options['summary'] = 1 
    #solver.options['SumName'] = f'summary_tree_{tree}'
    
    CPU_start_time = time.perf_counter()
    constant_fit_tree, rmse, r_squared, objective_value = constant_fit(es, cst_count, X, y, solver, tree)
    CPU_end_time = time.perf_counter()

    #Calculation of BIC post hoc
    #Penalizing depth
    p = depth
    n = len(y)
    range_y = np.max(y)-np.min(y)  #constant_fit returns the normalized rmse
    ssr = ( (rmse*range_y) **2) *n
    bic = n*log(ssr/n + 1e-6) + p*log(n)
    print('BIC=', bic)    
    
    results.append((tree, constant_fit_tree, rmse, p, bic))
    CPU_time = CPU_end_time - CPU_start_time
    print('CPU time to evaluate tree=', CPU_time)
    
    #print('objective_value', objective_value)
    if objective_value <= cutoff:
        cutoff = objective_value
        print('The new cutoff is', cutoff)
        solver.options['CutOff'] = cutoff + 1e-6
        
    print()
    if rmse < min_rmse:
        min_rmse = rmse
        best_model_dataset_name = (tree, constant_fit_tree, rmse, r_squared, p, bic)
        
CPU_total_end_time = time.perf_counter()
CPU_total_time = CPU_total_end_time - CPU_total_start_time
print('CPU total time to evaluate all trees=', CPU_total_time)

# training_time = "testing"
ending_time = time.perf_counter()
training_time = ending_time - starting_time

with open(file_path_constant_fit_trees, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(best_model_dataset_name)  #Write the best model first
    for result in results:
        writer.writerow(result)

with open(f'best_models_{dataset_name}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row with column names
    writer.writerow(['Output', 'Tree', 'Expression', 'nRMSE', 'R2', 'train_time', 'penalty_coef', 'BIC'])
    
    # Write the row
    writer.writerow([dataset_name, best_model_dataset_name[0], best_model_dataset_name[1], best_model_dataset_name[2],
                     best_model_dataset_name[3], training_time, p, bic])


# #### Step 7: Check BARON's output files
# 
# During step 6, the BARON solver created several files for each run. These files are saved in the user's local computer at the locations printed on the screen above.
# 
# For each tree expression, a summary file is created that contains information about the optimization problem that BARON solved. The summary files are named as 'summary_tree_{number_of_tree}'. 
# 
# For example, by defining the solver_output_file as 'summary_tree_3', and by uncommenting the commands in the cell below, the user is able to open the file using Notepad and/or to print the contents of the summary file on the screen. The information displayed will concern the tree expression '3'. 
# 
# Additionally, the user can replace the name 'summary_tree_{number_of_tree}' with any of the file locations above to see the contents of the BARON's log, solution or problem files.

# In[8]:


#solver_output_file = 'summary_tree_3'          # The solver output file path

#os.startfile(solver_output_file)               # Open the file in its associated program (notepad)

#with open(solver_output_file, 'r') as file:
    #content = file.read()
#print(content)                                 # Display the contents on the screen


# ### 3.2. Evaluating the model
# 
# In this section we will make predictions on the training points using the symbolic regression model that was created for the output 'liquid_outlet_flow_vol'. In this way we will create the training parity plot.
# 
# In order to further evaluate our model, we will make predictions on a testing set of 1,000 datapoints. The data can be found in the csv file '3_4_simulation_data_1000_test.csv'. We will calculate the testing performance metrics and create the testing parity plot. As mentioned above the combined data (training and testing points) have been min-max scaled between the numbers 3 and 4.

# #### Step 1: Import the needed packages

# In[9]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, max_error
from sklearn.metrics import r2_score


# #### Step 2: Import the SR surrogate model from the 'best_models.csv' file

# In[10]:


# Load the csv file
df_models = pd.read_csv(f'best_models_{dataset_name}.csv')
print(df_models)

# Extract the expression from the desired cell 
expression = df_models.loc[df_models['Output'].str.strip() == f'{k}_{output_name}', 'Expression'].values[0]
#print('Expression:',expression)

# Replace the placeholders in the expression
expression = expression.replace('X[:,2-1]', 'X2')
expression = expression.replace('X[:,1-1]', 'X1')
#print('Expression: y =',expression)

def SR_model(X1,X2):
    return eval(expression)

import sympy as sp
# Define symbolic variables
X1, X2 = sp.symbols('X1 X2')
# Parse the expression into sympy
sympy_expr = sp.sympify(expression)

# Simplify the expression
simplified_expr = sp.simplify(sympy_expr)
# Print the simplified expression
print('Expression: y =', simplified_expr)


# #### Step 3: Predict $\hat{y}$ values for the training set using the SR surrogate model

# In[11]:


# Applying the SR_model function to each pair of values
y_pred = np.array([SR_model(X[i,0], X[i,1]) for i in range(len(X))])
#print(y_pred)

# Stack the actual and predicted values column-wise
predicted_train = np.column_stack((y, y_pred))
# Save the numpy array to a CSV file
np.savetxt(f'train_results_{k}_{output_name}.csv', 
           predicted_train, delimiter=',', header=f'{output_name},{output_name}_pred', comments='', fmt='%f')

#Plot the training parity plot
z=[3,4]
w=[3,4]
fig, ax= plt.subplots(figsize=(6, 5))
ax.scatter(y, y_pred, alpha=0.3) 
ax.plot(z,w, 'k--') #y=x line
ax.set_xlabel('Actual', fontsize=12)
ax.set_ylabel('Predicted', fontsize=12)
ax.set_title(f'{output_name}', fontsize=10)
plt.tight_layout()
plt.savefig(f'parity_train_{output_name}_{k}.png', format='png')  # Save plot as a PNG file
plt.show()
plt.close(fig)  # Close the figure to free memory

r2_train = r2_score(y,y_pred)
print(f'R2 training: {r2_train}')

mse_train = mean_squared_error(y,y_pred)
print(f'MSE training: {mse_train}')

rmse_train = sqrt(mse_train)
print(f'RMSE training: {rmse_train}')   


# #### Step 4: Read the csv file with the testing set

# In[12]:


columns = ['liquid_inlet_conc_mol_comp_H2SO4','solid_inlet_flow_mass',
           f'{output_name}'] 

# read in our csv testing data
df_testing = pd.read_csv('3_4_simulation_data_1000_test.csv', usecols=columns)

# separate the data into inputs and outputs
inputs = ['liquid_inlet_conc_mol_comp_H2SO4','solid_inlet_flow_mass']

actual_outputs= [f'{output_name}']

dfin_testing = df_testing[inputs]
dfout_testing = df_testing[actual_outputs]

#Convert the pandas dataframes to numpy arrays
X_test = dfin_testing.to_numpy()
y_test = dfout_testing.to_numpy()

#scaler = MinMaxScaler((3,4))
#X_test = scaler.fit_transform(X_test)
#y_test = scaler.fit_transform(y_test.reshape(-1, 1))


# #### Step 5: Predict $\hat{y}$ values for the testing set using the SR surrogate model

# In[13]:


# Applying the SR_model function to each pair of values
y_pred_test = np.array([SR_model(X_test[i,0], X_test[i,1]) for i in range(len(X_test))])

# Stack the actual and predicted values column-wise
predicted_test = np.column_stack((y_test, y_pred_test))
# Save the numpy array to a CSV file
np.savetxt(f'test_results_{k}_{output_name}.csv', 
           predicted_test, delimiter=',', header=f'{output_name},{output_name}_pred', comments='', fmt='%f')

#Plot the training parity plot
z=[3,4]
w=[3,4]
fig, ax= plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred_test, alpha=0.3) 
ax.plot(z,w, 'k--') #y=x line
ax.set_xlabel('Actual', fontsize=12)
ax.set_ylabel('Predicted', fontsize=12)
ax.set_title(f'{output_name}', fontsize=10)
plt.tight_layout()
plt.savefig(f'parity_test_{output_name}_{k}.png', format='png')  # Save plot as a PNG file
plt.show()
plt.close(fig)  # Close the figure to free memory

r2_test = r2_score(y_test,y_pred_test)
print(f'R2 test: {r2_test}')

mse_test = mean_squared_error(y_test,y_pred_test)
print(f'MSE test: {mse_test}')

rmse_test = sqrt(mse_test)
print(f'RMSE test: {rmse_test}')   

