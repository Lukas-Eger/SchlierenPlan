

## Detection of local changes in the refractive index in additive manufactured devices using digital image processing
This work investigates how local changes in the refractive index of additively manufactured optical components can be detected, even during the manufacturing process. Our approach employs the shadowgraph method in combination with image processing to achieve this goal. The test sample was manufactured using multijet modelling. Unfortunately, this approach does not achieve the desired result because the surface roughness of the test sample is insufficient for this method. Further testing with other samples is required to evaluate the suitability of the shadowgraph method for detecting local changes in refractive index.<br>
This repo contains the **Python code** for the **image processing pipeline**, as well as the resulting **images** from the **test setup.**

We recommend using our virtual environment created with Anaconda to avoid conflicts between different version of packages.<br>
Importing the environment and then activating it with Anaconda works as follows:
```
		conda create --name <env_name> -f env_sp311.yml

		conda activate <env_name>
```
Our image processing pipline is implemented in */example/example.py*. It uses the specially developed SPImageProcessing package in */src/SPImageProcessing.* Which images or parameters are accessed in */img* can be set in the example script. The parameters for images processing can be configured in */example/config.py*.<br>
The resulting images of the final test setup are stored in */img/T04*.<br>
![alt text](./img/T03/T03_Test_Setup.jpg =385x166 "Test Setup 3")
