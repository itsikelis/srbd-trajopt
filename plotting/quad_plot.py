import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("quad_trot_step.csv")


# PLOT BODY POSITIONS
def plot_body_pos():
    indices = df["index"]
    x = df["x_b"]
    y = df["y_b"]
    z = df["z_b"]

    # Plotting the data
    plt.plot(indices.values / 100, x.values, linestyle="-", color="#4C86A8", label="x")
    plt.plot(indices.values / 100, y.values, linestyle="-", color="#F9C80E", label="y")
    plt.plot(
        indices.values / 100, z.values, linestyle="-", color="#c63c3c", label="theta"
    )

    # plt.axhline(y=-1.0, color="#4C86A8", linestyle="--", label="x=-1")
    # plt.axhline(y=-2.0, color="#F9C80E", linestyle="--", label="y=-2")
    # plt.axhline(y=0.0, color="#c63c3c", linestyle="--", label="theta=0")

    plt.title("Quadruped SRBD State Transition")
    plt.xlabel("t(s)")
    plt.ylabel("x(m),y(m),z(m)")
    plt.grid(True)
    plt.legend()
    plt.show()


# PLOT FEET POSITIONS
def plot_feet_pos():
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.180))

    x0 = df["x0"]
    y0 = df["y0"]
    z0 = df["z0"]

    indices = np.arange(0, x0.values.size)

    axs[0, 0].plot(
        indices / 100,
        x0.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[0, 0].plot(
        indices / 100,
        y0.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[0, 0].plot(
        indices / 100,
        z0.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[0, 0].set_title("Right Front Leg")
    axs[0, 0].set_ylabel("x_rf(m),y_rf(m),z_rf(m)")
    axs[0, 0].set_xlabel("t(s)")
    axs[0, 0].grid(True)

    x1 = df["x1"]
    y1 = df["y1"]
    z1 = df["z1"]

    axs[0, 1].plot(
        indices / 100,
        x1.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[0, 1].plot(
        indices / 100,
        y1.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[0, 1].plot(
        indices / 100,
        z1.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[0, 1].set_title("Left Front Leg")
    axs[0, 1].set_ylabel("x_lf(m),y_lf(m),z_lf(m)")
    axs[0, 1].set_xlabel("t(s)")
    axs[0, 1].grid(True)

    x2 = df["x2"]
    y2 = df["y2"]
    z2 = df["z2"]

    axs[1, 0].plot(
        indices / 100,
        x2.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[1, 0].plot(
        indices / 100,
        y2.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[1, 0].plot(
        indices / 100,
        z2.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[1, 0].set_title("Right Hind Leg")
    axs[1, 0].set_ylabel("x_rh(m),y_rh(m),z_rh(m)")
    axs[1, 0].set_xlabel("t(s)")
    axs[1, 0].grid(True)

    x3 = df["x3"]
    y3 = df["y3"]
    z3 = df["z3"]

    axs[1, 1].plot(
        indices / 100,
        x3.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[1, 1].plot(
        indices / 100,
        y3.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[1, 1].plot(
        indices / 100,
        z3.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[1, 1].set_title("Left Hind Leg")
    axs[1, 1].set_ylabel("x_lh(m),y_lh(m),z_lh(m)")
    axs[1, 1].set_xlabel("t(s)")
    axs[1, 1].grid(True)

    # # Set the same x and y limits for both subplots
    common_ylim = (
        min(
            min(x0),
            min(x1),
            min(x2),
            min(x3),
            min(y0),
            min(y1),
            min(y2),
            min(y3),
            min(z0),
            min(z1),
            min(z2),
            min(z3),
        ),
        max(
            max(x0),
            max(x1),
            max(x2),
            max(x3),
            max(y0),
            max(y1),
            max(y2),
            max(y3),
            max(z0),
            max(z1),
            max(z2),
            max(z3),
        ),
    )
    axs[0, 0].set_ylim(common_ylim)
    axs[0, 1].set_ylim(common_ylim)
    axs[1, 0].set_ylim(common_ylim)
    axs[1, 1].set_ylim(common_ylim)

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


# PLOT FEET FORCES
def plot_feet_forces():
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.180))

    x0 = df["fx0"]
    y0 = df["fy0"]
    z0 = df["fz0"]

    indices = np.arange(0, x0.values.size)

    axs[0, 0].plot(
        indices / 100,
        x0.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[0, 0].plot(
        indices / 100,
        y0.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[0, 0].plot(
        indices / 100,
        z0.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[0, 0].set_title("Right Front Force")
    axs[0, 0].set_ylabel("f_x(N),f_y(N),f_z(N)")
    axs[0, 0].set_xlabel("t(s)")
    axs[0, 0].grid(True)

    x1 = df["fx1"]
    y1 = df["fy1"]
    z1 = df["fz1"]

    axs[0, 1].plot(
        indices / 100,
        x1.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[0, 1].plot(
        indices / 100,
        y1.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[0, 1].plot(
        indices / 100,
        z1.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[0, 1].set_title("Left Front Force")
    axs[0, 1].set_ylabel("f_x(N),f_y(N),f_z(N)")
    axs[0, 1].set_xlabel("t(s)")
    axs[0, 1].grid(True)

    x2 = df["fx2"]
    y2 = df["fy2"]
    z2 = df["fz2"]

    axs[1, 0].plot(
        indices / 100,
        x2.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[1, 0].plot(
        indices / 100,
        y2.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[1, 0].plot(
        indices / 100,
        z2.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[1, 0].set_title("Right Hind Force")
    axs[1, 0].set_ylabel("f_x(N),f_y(N),f_z(N)")
    axs[1, 0].set_xlabel("t(s)")
    axs[1, 0].grid(True)

    x3 = df["fx3"]
    y3 = df["fy3"]
    z3 = df["fz3"]

    axs[1, 1].plot(
        indices / 100,
        x3.values,
        linestyle="-",
        color="#4C86A8",
        label="x",
    )
    axs[1, 1].plot(
        indices / 100,
        y3.values,
        linestyle="-",
        color="#F9C80E",
        label="y",
    )
    axs[1, 1].plot(
        indices / 100,
        z3.values,
        linestyle="-",
        color="#c63c3c",
        label="z",
    )
    axs[1, 1].set_title("Left Hind Force")
    axs[1, 1].set_ylabel("f_x(N),f_y(N),f_z(N)")
    axs[1, 1].set_xlabel("t(s)")
    axs[1, 1].grid(True)

    # # Set the same x and y limits for both subplots
    common_ylim = (
        min(
            min(x0),
            min(x1),
            min(x2),
            min(x3),
            min(y0),
            min(y1),
            min(y2),
            min(y3),
            min(z0),
            min(z1),
            min(z2),
            min(z3),
        ),
        max(
            max(x0),
            max(x1),
            max(x2),
            max(x3),
            max(y0),
            max(y1),
            max(y2),
            max(y3),
            max(z0),
            max(z1),
            max(z2),
            max(z3),
        ),
    )
    axs[0, 0].set_ylim(common_ylim)
    axs[0, 1].set_ylim(common_ylim)
    axs[1, 0].set_ylim(common_ylim)
    axs[1, 1].set_ylim(common_ylim)

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


# plot_body_pos()
plot_feet_pos()
# plot_feet_forces()
