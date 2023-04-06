import requests
import json

sample =    {'age':55,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"Doctorate",
                'education_num':16,
                'marital_status':"Separated",
                'occupation':"Exec-managerial",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Male",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':50,
                'native_country':"United-States"
            }

data = json.dumps(sample)
response = requests.post('https://project3-api-mse7.onrender.com/predictions', data=data)

print(response.status_code)
print(response.json())