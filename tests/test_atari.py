from cor_rl import factories
from cor_rl import atari


def test_frame_preprocessing():

    env = factories.atari_factory.get("SpaceInvaders")
    prev_frame, _ = env.reset(seed=101)
    frame, reward, terminated, trunc, info = env.step(0)

    comp_frame = atari.preprocess_frames(frame, prev_frame)
    # Test that frame is correct size
    assert comp_frame.shape == (84, 84)
    # Test that frame has correct scale
    assert comp_frame.max() <= 1
    # Test that frame has content
    assert comp_frame.std() > 27.0/255
