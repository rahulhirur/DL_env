@echo Initiating Exercise 0 Test
@echo Author - Rahul Jagadish Hirur Sachin Bharadwaj M
@echo Date - %date% Time - %time%

@echo Press any key to initate testing - exercise 1
@pause>null
@echo off
cd /D "%~dp0"

E:\Computational_Engineering\Subjects\Deep_Learning\DL_env\venv\Scripts\python.exe NeuralNetworkTests.py Bonus
@echo on
@echo Execution of test complete

@echo Press any key to exit
@pause>null