# 🌶 Flask

This section contains the Flask Restful API that I developed for the deployment process of a machine learning model during my participation in Bangkit Academy 2023.

### Endpoint

#### POST

```
  POST/
```

##### Input

| Key                        | Info     | Data Type |
| :------------------------- | :------- | :-------- |
| `Pregnancies`              | Required | float     |
| `Glucose`                  | Required | float     |
| `BloodPressure`            | Required | float     |
| `SkinThickness`            | Required | float     |
| `Insulin`                  | Required | float     |
| `BMI`                      | Required | float     |
| `DiabetesPedigreeFunction` | Required | float     |
| `Age`                      | Required | float     |

##### Output

| Key          | Data Type |
| :----------- | :-------- |
| `advice`     | string    |
| `prediction` | boolean   |
| `probabilty` | string    |
