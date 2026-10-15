
import pandas as pd

schemes = pd.read_csv("data/schemes.csv")

print("GOVERNMENT SCHEME MATCHING")
print("-" * 35)

age = int(input("Enter age: "))
income = float(input("Enter annual family income (Rs): "))

category = input(
    "Enter category (SC/OBC/EWS/DNT/Safai Mitra/Other): "
).strip().upper()

trade = input(
    "Are you working in one of the 18 traditional trades? (yes/no): "
).strip().lower()

print("\nPOTENTIAL SCHEME MATCHES")
print("-" * 35)

# Check PM-DAKSH
daksh_category = category in ["SC", "OBC", "EWS", "DNT", "SAFAI MITRA"]
daksh_income_ok = (
    category in ["SC", "DNT", "SAFAI MITRA"]
    or (
        category in ["OBC", "EWS"]
        and income < 300000
    )
)

if 18 <= age <= 45 and daksh_category and daksh_income_ok:
    print("PM-DAKSH: Potential match")
    print("Benefits: Skill training and related support")
    print("https://www.myscheme.gov.in/schemes/pm-daksh")
else:
    print("PM-DAKSH: Basic conditions not matched")

# Check PM Vishwakarma
if age >= 18 and trade == "yes":
    print("\nPM Vishwakarma: Potential match")
    print("Benefits: Training, toolkit and other support")
    print("https://www.myscheme.gov.in/schemes/pmv")
else:
    print("\nPM Vishwakarma: Basic conditions not matched")

print("\nFinal eligibility must be verified officially.")

