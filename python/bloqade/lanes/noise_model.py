from typing import TYPE_CHECKING, Any

from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.transform import SimpleLogicalNoiseModel, SimpleNoiseModel

if TYPE_CHECKING:
    from bloqade.cirq_utils.noise.model import (
        GeminiNoiseModelABC,
    )

PAIRED_KEYS = [
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]


def generate_simple_noise_model(
    noise_model: "GeminiNoiseModelABC | None" = None,
    loss: bool = True,
) -> SimpleNoiseModel:
    """Generate a physical noise model from a bloqade-circuit noise model.

    Args:
        noise_model: The bloqade-circuit noise model to use. Defaults to None.
        loss: Whether to include loss in the noise model. Defaults to True.

    Returns:
        A simple noise model for physical gate/move noise insertion.
    """
    from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    cz_unpaired_loss_prob = noise_model.cz_unpaired_loss_prob
    cz_unpaired_gate_px = noise_model.cz_unpaired_gate_px
    cz_unpaired_gate_py = noise_model.cz_unpaired_gate_py
    cz_unpaired_gate_pz = noise_model.cz_unpaired_gate_pz

    @squin.kernel
    def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("CZ Unpaired Noise")
        squin.broadcast.single_qubit_pauli_channel(
            cz_unpaired_gate_px, cz_unpaired_gate_py, cz_unpaired_gate_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(cz_unpaired_loss_prob, qubits)

    mover_px = noise_model.mover_px
    mover_py = noise_model.mover_py
    mover_pz = noise_model.mover_pz
    move_lost_prob = noise_model.move_loss_prob

    @squin.kernel
    def lane_noise(qubit: qubit.Qubit):
        debug.info("Lane Noise")
        squin.single_qubit_pauli_channel(mover_px, mover_py, mover_pz, qubit)
        if loss:
            squin.qubit_loss(move_lost_prob, qubit)

    sitter_px = noise_model.sitter_px
    sitter_py = noise_model.sitter_py
    sitter_pz = noise_model.sitter_pz
    sit_loss_prob = noise_model.sit_loss_prob

    @squin.kernel
    def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("Idle Noise")
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, qubits)

    cz_paired_error_dict = noise_model.cz_paired_error_probabilities
    if cz_paired_error_dict is None:
        raise ValueError("CZ paired error probabilities must be provided.")

    cz_paired_error_probabilities = ilist.IList(
        [cz_paired_error_dict[k] for k in PAIRED_KEYS]
    )

    cz_unpaired_loss_prob = noise_model.cz_gate_loss_prob

    @squin.kernel
    def cz_paired_noise(
        controls: ilist.IList[qubit.Qubit, Any], targets: ilist.IList[qubit.Qubit, Any]
    ):
        debug.info("CZ Paired Noise")
        squin.broadcast.two_qubit_pauli_channel(
            cz_paired_error_probabilities, controls, targets
        )

        def pair_qubit(i: int):
            return ilist.IList([controls[i], targets[i]])

        if loss:
            groups = ilist.map(pair_qubit, ilist.range(len(controls)))
            squin.broadcast.correlated_qubit_loss(cz_unpaired_loss_prob, groups)

    local_px = noise_model.local_px
    local_py = noise_model.local_py
    local_pz = noise_model.local_pz
    local_loss_prob = noise_model.local_loss_prob

    @squin.kernel
    def local_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Local Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    @squin.kernel
    def local_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Local Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    global_px = noise_model.global_px
    global_py = noise_model.global_py
    global_pz = noise_model.global_pz
    global_loss_prob = noise_model.global_loss_prob

    @squin.kernel
    def global_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Global Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(global_loss_prob, qubits)

    @squin.kernel
    def global_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Global Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(global_loss_prob, qubits)

    return SimpleNoiseModel(
        lane_noise=lane_noise,
        idle_noise=idle_noise,
        cz_unpaired_noise=cz_unpaired_noise,
        cz_paired_noise=cz_paired_noise,
        global_rz_noise=global_rz_noise,
        local_rz_noise=local_rz_noise,
        global_r_noise=global_r_noise,
        local_r_noise=local_r_noise,
    )


def generate_logical_noise_model(
    noise_model: "GeminiNoiseModelABC | None" = None,
    loss: bool = True,
) -> SimpleLogicalNoiseModel:
    """Generate a logical noise model with initialization kernels.

    Creates a physical noise model and adds Steane [[7,1,3]] clean and noisy
    initialization kernels, all derived from the same source parameters.

    Args:
        noise_model: The bloqade-circuit noise model to use. Defaults to None.
        loss: Whether to include loss channels. Defaults to True.

    Returns:
        A logical noise model with gate/move noise and both initialization kernels.
    """
    from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    physical = generate_simple_noise_model(noise_model, loss=loss)

    from bloqade.lanes.arch.gemini.logical.upstream import steane7_initialize_with_noise

    clean_init, noisy_init = steane7_initialize_with_noise(
        local_px=noise_model.local_px,
        local_py=noise_model.local_py,
        local_pz=noise_model.local_pz,
        local_loss_prob=noise_model.local_loss_prob,
        mover_px=noise_model.mover_px,
        mover_py=noise_model.mover_py,
        mover_pz=noise_model.mover_pz,
        move_loss_prob=noise_model.move_loss_prob,
        sitter_px=noise_model.sitter_px,
        sitter_py=noise_model.sitter_py,
        sitter_pz=noise_model.sitter_pz,
        sit_loss_prob=noise_model.sit_loss_prob,
        loss=loss,
    )

    return SimpleLogicalNoiseModel.from_simple(
        physical,
        logical_initialize_clean=clean_init,
        logical_initialize_noisy=noisy_init,
    )

def generate_coherent_noise_model(
    noise_model: "GeminiNoiseModelABC | None" = None,
    loss: bool = True,
) -> SimpleNoiseModel:
    """Generate a noise model combining coherent unitary errors with Pauli errors.

    For each single-qubit gate with rotation angle Omega (in radians), a coherent
    error unitary is applied immediately after the ideal gate, followed by the
    Pauli error channel. The coherent error is represented as an arbitrary u3
    rotation (ZYZ convention) whose angles scale linearly with Omega:

        u3(epsilon_theta * Omega, epsilon_phi * Omega, epsilon_lambda * Omega)

    The full noisy channel per gate is therefore:

        E(rho) = P( U_err . U . rho . U† . U_err† )

    where U is the ideal gate, U_err = u3(epsilon_theta * Omega, ...), and P is
    the asymmetric Pauli channel. Both coherent and Pauli parameters are read
    from noise_model, consistent with how generate_simple_noise_model works.

    Local and global coherent errors are parameterized separately since they
    correspond to physically distinct error sources (Raman laser miscalibration
    vs global beam miscalibration).

    The rotation angle Omega is passed directly by the compiler into the noise
    kernel as `rotation_angle` (in radians) for all single-qubit gate types
    (LocalR, GlobalR, LocalRz, GlobalRz).

    Coherent errors are applied only to single-qubit rotation gates. Lane, idle,
    and CZ kernels are identical to generate_simple_noise_model — coherent errors
    on CZ gates are a planned future extension.

    Args:
        noise_model:    Gemini noise model supplying both Pauli error probabilities
                        and coherent error parameters. Defaults to
                        GeminiOneZoneNoiseModel(). Coherent error parameters are
                        zero by default, recovering generate_simple_noise_model
                        behaviour when not set.
        loss:           Whether to include qubit loss channels. Defaults to True.

    Returns:
        A SimpleNoiseModel with coherent + Pauli single-qubit gate noise.
    """
    from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    # -------------------------------------------------------------------------
    # Lane, idle, and CZ kernels — Pauli errors only, unchanged from
    # generate_simple_noise_model.
    # -------------------------------------------------------------------------

    cz_unpaired_loss_prob = noise_model.cz_unpaired_loss_prob
    cz_unpaired_gate_px = noise_model.cz_unpaired_gate_px
    cz_unpaired_gate_py = noise_model.cz_unpaired_gate_py
    cz_unpaired_gate_pz = noise_model.cz_unpaired_gate_pz

    @squin.kernel
    def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("CZ Unpaired Noise")
        squin.broadcast.single_qubit_pauli_channel(
            cz_unpaired_gate_px, cz_unpaired_gate_py, cz_unpaired_gate_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(cz_unpaired_loss_prob, qubits)

    mover_px = noise_model.mover_px
    mover_py = noise_model.mover_py
    mover_pz = noise_model.mover_pz
    move_lost_prob = noise_model.move_loss_prob

    @squin.kernel
    def lane_noise(qubit: qubit.Qubit):
        debug.info("Lane Noise")
        squin.single_qubit_pauli_channel(mover_px, mover_py, mover_pz, qubit)
        if loss:
            squin.qubit_loss(move_lost_prob, qubit)

    sitter_px = noise_model.sitter_px
    sitter_py = noise_model.sitter_py
    sitter_pz = noise_model.sitter_pz
    sit_loss_prob = noise_model.sit_loss_prob

    @squin.kernel
    def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("Idle Noise")
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, qubits)

    cz_paired_error_dict = noise_model.cz_paired_error_probabilities
    if cz_paired_error_dict is None:
        raise ValueError("CZ paired error probabilities must be provided.")

    cz_paired_error_probabilities = ilist.IList(
        [cz_paired_error_dict[k] for k in PAIRED_KEYS]
    )

    cz_unpaired_loss_prob = noise_model.cz_gate_loss_prob

    @squin.kernel
    def cz_paired_noise(
        controls: ilist.IList[qubit.Qubit, Any], targets: ilist.IList[qubit.Qubit, Any]
    ):
        debug.info("CZ Paired Noise")
        squin.broadcast.two_qubit_pauli_channel(
            cz_paired_error_probabilities, controls, targets
        )

        def pair_qubit(i: int):
            return ilist.IList([controls[i], targets[i]])

        if loss:
            groups = ilist.map(pair_qubit, ilist.range(len(controls)))
            squin.broadcast.correlated_qubit_loss(cz_unpaired_loss_prob, groups)

    # -------------------------------------------------------------------------
    # Single-qubit gate noise model — coherent error followed by Pauli error.
    #
    # Coherent error parameters are read from noise_model alongside the Pauli
    # rates, following the same pattern as generate_simple_noise_model.
    #
    # The coherent error u3(theta, phi, lam) uses the ZYZ convention:
    # u3(theta, phi, lam) = Rz(phi) . Ry(theta) . Rz(lam)
    # All three angles scale linearly with the gate rotation_angle
    # -------------------------------------------------------------------------

    local_px = noise_model.local_px
    local_py = noise_model.local_py
    local_pz = noise_model.local_pz
    local_loss_prob = noise_model.local_loss_prob
    local_epsilon_theta  = noise_model.local_coherent_epsilon_theta
    local_epsilon_phi    = noise_model.local_coherent_epsilon_phi
    local_epsilon_lambda = noise_model.local_coherent_epsilon_lambda

    @squin.kernel
    def local_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Local Gate Coherent + Pauli Noise")
        squin.broadcast.u3(
            local_epsilon_theta  * rotation_angle,
            local_epsilon_phi    * rotation_angle,
            local_epsilon_lambda * rotation_angle,
            qubits,
        )
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    @squin.kernel
    def local_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Local Rz Coherent + Pauli Noise")
        squin.broadcast.u3(
            local_epsilon_theta  * rotation_angle,
            local_epsilon_phi    * rotation_angle,
            local_epsilon_lambda * rotation_angle,
            qubits,
        )
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    global_px = noise_model.global_px
    global_py = noise_model.global_py
    global_pz = noise_model.global_pz
    global_loss_prob = noise_model.global_loss_prob
    global_epsilon_theta  = noise_model.global_coherent_epsilon_theta
    global_epsilon_phi    = noise_model.global_coherent_epsilon_phi
    global_epsilon_lambda = noise_model.global_coherent_epsilon_lambda

    @squin.kernel
    def global_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Global Gate Coherent + Pauli Noise")
        squin.broadcast.u3(
            global_epsilon_theta  * rotation_angle,
            global_epsilon_phi    * rotation_angle,
            global_epsilon_lambda * rotation_angle,
            qubits,
        )
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(global_loss_prob, qubits)

    @squin.kernel
    def global_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Global Rz Coherent + Pauli Noise")
        squin.broadcast.u3(
            global_epsilon_theta  * rotation_angle,
            global_epsilon_phi    * rotation_angle,
            global_epsilon_lambda * rotation_angle,
            qubits,
        )
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(global_loss_prob, qubits)

    return SimpleNoiseModel(
        lane_noise=lane_noise,
        idle_noise=idle_noise,
        cz_unpaired_noise=cz_unpaired_noise,
        cz_paired_noise=cz_paired_noise,
        global_rz_noise=global_rz_noise,
        local_rz_noise=local_rz_noise,
        global_r_noise=global_r_noise,
        local_r_noise=local_r_noise,
    )
