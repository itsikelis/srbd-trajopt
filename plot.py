import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("build/output.csv")

indices = df["index"]
x = df["fx1"]
y = df["fy1"]
z = df["fz1"]

# Plotting the data
# plt.plot(indices, x, marker='o', linestyle='-', color='b')
plt.plot(indices.values, x.values, linestyle="-", color="#4C86A8", label="X")
plt.plot(indices.values, y.values, linestyle="-", color="#F9C80E", label="Y")
plt.plot(indices.values, z.values, linestyle="-", color="#c63c3c", label="Z")

# Plot f as zero order hold
# plt.plot(indices, f, drawstyle='steps', linestyle='-', color='g')

# plt.axhline(y=3.14, color="r", linestyle="--", label="Y = 3.14")
# plt.axhline(y=0.0, color="b", linestyle="--", label="Y = 3.14")

# Plot f as zero order hold
# plt.plot(
#     indices.values,
#     u_r.values,
#     drawstyle="steps",
#     linestyle="-",
#     color="#7EBC89",
#     label="Right Thrust",
# )

# plt.plot(
#     indices.values,
#     u_l.values,
#     drawstyle="steps",
#     linestyle="-",
#     color="#474350",
#     label="Left Thrust",
# )


plt.title("State Transition and Controls")
plt.xlabel("Index")
plt.ylabel("Y Label")
plt.grid(True)
plt.legend()
plt.show()
