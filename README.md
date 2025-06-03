# Call Me Fragrant Acupuncture Master

## Project Motivation
As CS students spending long hours coding, we often experience:
- Headaches
- Wrist pain
- Eye strain
  
NYCU campus is located on a hill, leading to frequent:
- Foot and leg pain from walking between classes

Many assignments and exams may also lead to :
- Stress
- Depression

These discomforts inspired us to develop an AI agent that:
- Recommends suitable acupoints and essential oils
- Shows the user where the acupoints are

## System Design
<img width="1129" alt="Screenshot 2025-06-03 at 2 20 51 PM" src="https://github.com/user-attachments/assets/2e811888-9fff-44b8-9876-116feca09546" />


## How to Run
1. Get all the files from this repository
```
git clone https://github.com/hoho56422/AI_final_acupressure.git
```
2. Download trained model in `pages\trained_model\download_url.txt`, and put the downloaded files under `trained_model/` folder
3. Make sure all the packages in `requirement.txt` are downloaded in your environment.
   For your reference, this system is ran under Python 3.11.7, but it should be able to run under any environment
4. Run `home.py` by typing
```
streamlit run home.py
```
