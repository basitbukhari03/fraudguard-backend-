import requests

url = "http://127.0.0.1:5000/predict"

# Legit transaction features
legit_features = [
    0.0, -1.359807, -0.072781, 2.536347, 1.378155,
    -0.338321, 0.462388, 0.239599, 0.098698, 0.363787,
    0.090794, -0.551600, -0.617801, -0.991390, -0.311169,
    1.468177, -0.470401, 0.207971, 0.025791, 0.403993,
    0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
    0.128539, -0.189115, 0.133558, -0.021053, 149.62
]

# Fraud transaction features
fraud_features = [
    406.0, -2.312227, 1.951992, -1.609851, 3.997906,
    -0.522188, -1.426545, -2.537387, 1.391657, -2.770089,
    -2.772272, 3.202033, -2.899907, -0.595222, -4.289254,
    0.389724, -1.140747, -2.830056, -0.016822, 0.416956,
    0.126911, 0.517232, -0.035049, -0.465211, 0.320198,
    0.044519, 0.177840, 0.261145, -0.143276, 0.0
]

# Test Legit
legit_response = requests.post(url, json={"features": legit_features})
print("Legit Transaction Result:")
print(legit_response.json())

# Test Fraud
fraud_response = requests.post(url, json={"features": fraud_features})
print("\nFraud Transaction Result:")
print(fraud_response.json())
