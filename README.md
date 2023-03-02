# Guided transfer learning 
This repository implements our Guide transfer learning(GTL) approach (see [our article](https://www.researchgate.net/publication/367378102_Guided_transfer_learning_Helping_deep_networks_when_learning_gets_tough)).
We propose a new approach called Guided transfer learning, which involves assigning guiding parameters to each weight and bias in the network, allowing for a reduction in resources needed to train a network. Guided transfer learning enables the network to learn from a small amount of data and potentially has many applications in resource-efficient machine learning.

# Installation
1. Clone the repository
```bash
git clone https://github.com/RobotsGoMental/gtl_poc.git
cd gtl_poc
```
2. Create a new conda environment with Python 3.9
```bash
conda create --name gtl python=3.9
```
3. Activate the environment:
```bash
conda activate gtl
``` 
4. Install the required packages using the following command
```bash
pip install -r requirements.txt
``` 
# Usage
1. Navigate to the right directory 
```bash
cd gtl_poc
```
2. Run the following command to execute the script:
```bash
python all.py
```
3. Run the following command to start TensorBoard:
```bash
tensorboard --logdir logs --bind_all
```
4. Open http://localhost:6006 in your browser to access TensorBoard.
    
# Credits

This project was created by Danko Nikolić, Davor Andrić, Vjekoslav Nikolić

© 2023, Robots Go Mental, UG or its Affiliates. 

# License

This project is licensed under the [License Name]. 
