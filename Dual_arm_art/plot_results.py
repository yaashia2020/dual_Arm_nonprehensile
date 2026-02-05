import matplotlib.pyplot as plt
import numpy as np


def _maybe_plot(ax, t, data, label, color=None, style="-"):
    if data is None or len(data) == 0:
        return
    ax.plot(t, data, label=label, linestyle=style, color=color)


def _unpack(logs, key):
    entry = logs.get(key)
    if entry is None:
        return None, None
    if not (isinstance(entry, tuple) and len(entry) == 2):
        return None, None
    return entry


def _align_time_and_data(t, data):
    """
    Ensure t and data have the same first dimension length.
    Returns (t_aligned, data_aligned) or (None, None) if unusable.
    """
    if t is None or data is None:
        return None, None
    t = np.asarray(t).reshape(-1)
    data = np.asarray(data)
    if t.size == 0:
        return None, None
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n = min(t.shape[0], data.shape[0])
    if n <= 1:
        return None, None
    return t[:n], data[:n]


def plot_results(logs, robot_spec, show=True):
    """
    Plot logged signals in a generic way.

    Args:
        logs: dict with entries:
            - state: (t, X) numpy arrays
            - tau: (t, tau) numpy arrays
            - qv: (t, qv) numpy arrays
            - qr: (t, qr) numpy arrays
            - box: (t, box_state) numpy arrays
            - energy: (t, energy) numpy arrays
            - dsm: (t, dsm) numpy arrays
            - contact: (t, contact_forces) numpy arrays
            - link_pos: (t, xyz) numpy arrays
        robot_spec: RobotSpec
    """
    nq = int(getattr(robot_spec, "num_positions", 0) or 0)
    nv = int(getattr(robot_spec, "num_velocities", 0) or 0)
    U = int(getattr(robot_spec, "controlled_dofs", 0) or 0)
    dq_max = np.array(getattr(robot_spec, "dq_max", np.array([])), dtype=float) if hasattr(robot_spec, "dq_max") else np.array([])
    tau_max = np.array(getattr(robot_spec, "tau_max", np.array([])), dtype=float) if hasattr(robot_spec, "tau_max") else np.array([])
    q_min = np.array(getattr(robot_spec, "q_min", np.array([])), dtype=float) if hasattr(robot_spec, "q_min") else np.array([])
    q_max = np.array(getattr(robot_spec, "q_max", np.array([])), dtype=float) if hasattr(robot_spec, "q_max") else np.array([])
    controlled_indices = np.array(getattr(robot_spec, "controlled_indices", np.array([])), dtype=int) if hasattr(robot_spec, "controlled_indices") else np.array([])

    # Unpack logs
    t_x, X = _unpack(logs, "state")
    t_tau, tau = _unpack(logs, "tau")
    t_qv, qv = _unpack(logs, "qv")
    t_qr, qr = _unpack(logs, "qr")
    t_box, box = _unpack(logs, "box")
    t_energy, energy = _unpack(logs, "energy")
    t_dsm, dsm = _unpack(logs, "dsm")
    t_contact, contact = _unpack(logs, "contact")
    t_link, link_pos = _unpack(logs, "link_pos")
    t_link6, link6_pos = _unpack(logs, "link6_pos")
    t_hand, hand_pos = _unpack(logs, "hand_pos")
    t_tcp, tcp_pos = _unpack(logs, "tcp_pos")
    t_pad, pad_pos = _unpack(logs, "pad_pos")
    t_box_target, box_target_pos = _unpack(logs, "box_target_pos")
    t_ik_hand, ik_hand_pos = _unpack(logs, "ik_hand_pos")

    # Align everything (avoid length mismatch crashes)
    t_x, X = _align_time_and_data(t_x, X)
    t_tau, tau = _align_time_and_data(t_tau, tau)
    t_qv, qv = _align_time_and_data(t_qv, qv)
    t_qr, qr = _align_time_and_data(t_qr, qr)
    t_box, box = _align_time_and_data(t_box, box)
    t_energy, energy = _align_time_and_data(t_energy, energy)
    t_dsm, dsm = _align_time_and_data(t_dsm, dsm)
    t_contact, contact = _align_time_and_data(t_contact, contact)
    t_link, link_pos = _align_time_and_data(t_link, link_pos)
    t_link6, link6_pos = _align_time_and_data(t_link6, link6_pos)
    t_hand, hand_pos = _align_time_and_data(t_hand, hand_pos)
    t_tcp, tcp_pos = _align_time_and_data(t_tcp, tcp_pos)
    t_pad, pad_pos = _align_time_and_data(t_pad, pad_pos)
    t_box_target, box_target_pos = _align_time_and_data(t_box_target, box_target_pos)
    t_ik_hand, ik_hand_pos = _align_time_and_data(t_ik_hand, ik_hand_pos)

    # For most plots we want a common time base.
    # Prefer state time; otherwise fall back to qv/qr/tau if present.
    # NOTE: these are numpy arrays; don't use Python `or` (it triggers ambiguous truth-value errors).
    t_common = next(
        (t for t in [t_x, t_qv, t_qr, t_tau, t_box, t_energy, t_dsm, t_contact, t_link, t_hand] if t is not None),
        None,
    )
    if t_common is None:
        if show:
            print("plot_results: no usable logs to plot.")
        return

    # Joint positions/refs
    if X is not None and nq > 0:
        q = X[:, :nq]

        # Align qv/qr to state time by truncation (same approach test_erg implicitly used)
        qv_plot = qv[:, :nq] if qv is not None else None
        qr_plot = qr[:, :nq] if qr is not None else None

        n = q.shape[0]
        if qv_plot is not None:
            n = min(n, qv_plot.shape[0])
        if qr_plot is not None:
            n = min(n, qr_plot.shape[0])
        t_plot = t_x[:n] if t_x is not None else t_common[:n]
        q = q[:n]
        if qv_plot is not None:
            qv_plot = qv_plot[:n]
        if qr_plot is not None:
            qr_plot = qr_plot[:n]

        # Joint positions (controlled joints only: typically the 7 arm joints)
        if controlled_indices.size > 0 and U > 0:
            joint_ids = controlled_indices[:U].tolist()
            nplot = len(joint_ids)
        else:
            joint_ids = list(range(nq))
            nplot = nq

        fig_q, axs_q = plt.subplots(nplot, 1, figsize=(12, 2.5 * nplot), sharex=True)
        if nplot == 1:
            axs_q = [axs_q]
        fig_q.suptitle("Joint Positions (controlled joints) with limits")

        for i in range(nplot):
            j = joint_ids[i]
            _maybe_plot(axs_q[i], t_plot, q[:, j], f"q[{j}]", style="-")
            if qv_plot is not None and j < qv_plot.shape[1]:
                _maybe_plot(axs_q[i], t_plot, qv_plot[:, j], f"q_v[{j}]", style="--")
            if qr_plot is not None and j < qr_plot.shape[1]:
                _maybe_plot(axs_q[i], t_plot, qr_plot[:, j], f"q_r[{j}]", style=":")

            # Limits from setup (RobotSpec.q_min/q_max are per controlled DOF index i)
            if i < q_min.shape[0] and i < q_max.shape[0]:
                axs_q[i].axhline(y=float(q_min[i]), color="red", linestyle="--", linewidth=1.0, label="q_min")
                axs_q[i].axhline(y=float(q_max[i]), color="red", linestyle="--", linewidth=1.0, label="q_max")

            axs_q[i].grid(True)
            axs_q[i].legend(loc="upper right", fontsize=8)

        axs_q[-1].set_xlabel("Time [s]")
        fig_q.tight_layout(rect=[0, 0.03, 1, 0.98])

        # Joint velocities if present in state (first min(nq, nv) entries)
        if nv > 0 and X.shape[1] >= (nq + nv):
            qdot = X[:n, nq : nq + nv]
            nd = min(nq, qdot.shape[1])
            fig_dq, axs_dq = plt.subplots(nd, 1, figsize=(12, 2.5 * nd), sharex=True)
            if nd == 1:
                axs_dq = [axs_dq]
            fig_dq.suptitle("Joint Velocities")
            for i in range(nd):
                _maybe_plot(axs_dq[i], t_plot, qdot[:, i], f"dq[{i}]", style="-")
                if i < min(U, dq_max.shape[0]) and dq_max.size > 0:
                    axs_dq[i].plot(t_plot, dq_max[i] * np.ones_like(t_plot), "r--", label="dq_limit")
                    axs_dq[i].plot(t_plot, -dq_max[i] * np.ones_like(t_plot), "r--")
                axs_dq[i].grid(True)
                axs_dq[i].legend(loc="upper right", fontsize=8)
            axs_dq[-1].set_xlabel("Time [s]")
            fig_dq.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Torques
    if tau is not None:
        # Per-joint torque plot (similar to joint position layout)
        tau_plot = tau
        nt = tau_plot.shape[0]
        t_tau_plot = t_tau[:nt] if t_tau is not None else t_common[:nt]
        nj = min(nq if nq > 0 else tau_plot.shape[1], tau_plot.shape[1])
        if nj > 0:
            fig_tau, axs_tau = plt.subplots(nj, 1, figsize=(12, 2.5 * nj), sharex=True)
            if nj == 1:
                axs_tau = [axs_tau]
            fig_tau.suptitle("Joint Torques")
            for i in range(nj):
                _maybe_plot(axs_tau[i], t_tau_plot, tau_plot[:, i], f"tau[{i}]", style="-")
                if i < min(U, tau_max.shape[0]) and tau_max.size > 0:
                    axs_tau[i].plot(t_tau_plot, tau_max[i] * np.ones_like(t_tau_plot), "r--", label="tau_limit")
                    axs_tau[i].plot(t_tau_plot, -tau_max[i] * np.ones_like(t_tau_plot), "r--")
                axs_tau[i].grid(True)
                axs_tau[i].legend(loc="upper right", fontsize=8)
            axs_tau[-1].set_xlabel("Time [s]")
            fig_tau.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Energy
    if energy is not None:
        t_e = t_energy
        e = energy
        if t_e is not None and e is not None:
            fig_energy, axs_energy = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig_energy.suptitle("System Energy")
            _maybe_plot(axs_energy[0], t_e, e[:, 0], "Total Energy", color="blue", style="-")
            axs_energy[0].set_ylabel("Energy [J]")
            axs_energy[0].grid(True)
            axs_energy[0].legend(loc="upper right")
            if e.shape[1] > 2:
                _maybe_plot(axs_energy[1], t_e, e[:, 1], "Kinetic Energy", color="red", style="-")
                _maybe_plot(axs_energy[1], t_e, e[:, 2], "Potential Energy", color="green", style="-")
            elif e.shape[1] > 1:
                _maybe_plot(axs_energy[1], t_e, e[:, 1], "Energy[1]", style="-")
            axs_energy[1].set_ylabel("Energy [J]")
            axs_energy[1].grid(True)
            axs_energy[1].legend(loc="upper right")
            axs_energy[-1].set_xlabel("Time [s]")
            fig_energy.tight_layout(rect=[0, 0.03, 1, 0.98])

    # DSM (dynamic safety margin)
    if dsm is not None and t_dsm is not None:
        fig_dsm, ax_dsm = plt.subplots(figsize=(10, 3))
        _maybe_plot(ax_dsm, t_dsm, dsm[:, 0], "DSM", color="purple", style="-")
        ax_dsm.axhline(y=0.0, color="black", linestyle="--", alpha=0.5)
        ax_dsm.set_xlabel("Time [s]")
        ax_dsm.set_ylabel("DSM")
        ax_dsm.grid(True)
        ax_dsm.legend(loc="upper right")
        fig_dsm.tight_layout()

    # Contact forces
    if contact is not None:
        t_c = t_contact
        c = contact
        if t_c is not None and c is not None:
            fig_contact, axs_contact = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            fig_contact.suptitle("Contact Forces")
            labels = ["Contact Force X", "Contact Force Y", "Contact Force Z"]
            colors = ["red", "green", "blue"]
            m = min(3, c.shape[1])
            for i in range(m):
                _maybe_plot(axs_contact[i], t_c, c[:, i], labels[i], color=colors[i], style="-")
                axs_contact[i].set_ylabel(f"Force {['X','Y','Z'][i]} [N]")
                axs_contact[i].grid(True)
                axs_contact[i].legend(loc="upper right")
            axs_contact[-1].set_xlabel("Time [s]")
            fig_contact.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Link position
    if link_pos is not None:
        fig_l, axs_l = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["x", "y", "z"]
        for i in range(min(3, link_pos.shape[1])):
            _maybe_plot(axs_l[i], t_link, link_pos[:, i], labels[i])
            axs_l[i].grid(True)
            axs_l[i].legend()
        axs_l[-1].set_xlabel("time [s]")
        fig_l.tight_layout()

    # Panda hand position
    if hand_pos is not None:
        fig_h, axs_h = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["x", "y", "z"]
        for i in range(min(3, hand_pos.shape[1])):
            _maybe_plot(axs_h[i], t_hand, hand_pos[:, i], labels[i])
            axs_h[i].grid(True)
            axs_h[i].legend()
        axs_h[-1].set_xlabel("time [s]")
        fig_h.suptitle("panda_hand position (world)")
        fig_h.tight_layout(rect=[0, 0.03, 1, 0.98])

    # IK target position (world) vs panda_hand (debug)
    if box_target_pos is not None and t_box_target is not None:
        fig_t, axs_t = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["x", "y", "z"]
        for i in range(min(3, box_target_pos.shape[1])):
            _maybe_plot(axs_t[i], t_box_target, box_target_pos[:, i], f"target_{labels[i]}", style="--")
            if hand_pos is not None and t_hand is not None:
                _maybe_plot(axs_t[i], t_hand, hand_pos[:, i], f"hand_{labels[i]}", style="-")
            if ik_hand_pos is not None and t_ik_hand is not None:
                _maybe_plot(axs_t[i], t_ik_hand, ik_hand_pos[:, i], f"ik_fk_{labels[i]}", style=":")
            if link_pos is not None and t_link is not None:
                _maybe_plot(axs_t[i], t_link, link_pos[:, i], f"link7_{labels[i]}", style="-.")
            if link6_pos is not None and t_link6 is not None:
                _maybe_plot(axs_t[i], t_link6, link6_pos[:, i], f"link6_{labels[i]}", style=(0, (1, 1)))
            if tcp_pos is not None and t_tcp is not None:
                _maybe_plot(axs_t[i], t_tcp, tcp_pos[:, i], f"tcp_{labels[i]}", style=(0, (3, 1, 1, 1)))
            if pad_pos is not None and t_pad is not None:
                _maybe_plot(axs_t[i], t_pad, pad_pos[:, i], f"pad_{labels[i]}", style=(0, (5, 2)))
            axs_t[i].grid(True)
            axs_t[i].legend(loc="upper right")
        axs_t[-1].set_xlabel("time [s]")
        fig_t.suptitle("IK target vs panda_hand (actual) vs panda_hand(q_r FK)")
        fig_t.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Box position (from 13-state: quat(wxyz) + pos(xyz) + vel(??))
    if box is not None and t_box is not None and box.shape[1] >= 7:
        pos = box[:, 4:7]
        fig_box, axs_box = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig_box.suptitle("Movable Box Position (XYZ)")
        colors = ["red", "green", "blue"]
        xyz = ["X", "Y", "Z"]
        for i in range(3):
            _maybe_plot(axs_box[i], t_box, pos[:, i], f"Position {xyz[i]}", color=colors[i], style="-")
            axs_box[i].set_ylabel(f"{xyz[i]} Position [m]")
            axs_box[i].grid(True)
            axs_box[i].legend(loc="upper right")
        axs_box[-1].set_xlabel("Time [s]")
        fig_box.tight_layout(rect=[0, 0.03, 1, 0.98])

    # IK tracking performance plots intentionally removed (requested).

    # Energy/contact summary stats (mirrors tail prints from test_erg)
    if energy is not None and energy.shape[0] > 0:
        try:
            print("\n=== Energy and Contact Forces Summary ===")
            print(f"Final total energy: {float(energy[-1, 0]):.4f} J")
            if energy.shape[1] > 1:
                print(f"Final kinetic energy: {float(energy[-1, 1]):.4f} J")
            if energy.shape[1] > 2:
                print(f"Final potential energy: {float(energy[-1, 2]):.4f} J")
        except Exception:
            pass
    if contact is not None and contact.shape[0] > 0:
        try:
            mags = np.linalg.norm(contact[:, : min(3, contact.shape[1])], axis=1)
            print(f"Max contact force magnitude: {float(np.max(mags)):.4f} N")
            print(f"Average contact force magnitude: {float(np.mean(mags)):.4f} N")
        except Exception:
            pass

    if show:
        plt.show()

