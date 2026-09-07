from event_chain import SIX_YARD_BOX_DEPTH_M, is_goalkeeper_run_out


def test_run_out_requires_contact_beyond_six_yard_box():
    goal_line = 105.0
    assert is_goalkeeper_run_out(
        goal_line, goal_line - SIX_YARD_BOX_DEPTH_M - 0.01
    )
    assert not is_goalkeeper_run_out(
        goal_line, goal_line - SIX_YARD_BOX_DEPTH_M + 0.01
    )


def test_run_out_works_for_keeper_defending_the_other_goal():
    goal_line = 0.0
    assert is_goalkeeper_run_out(
        goal_line, goal_line + SIX_YARD_BOX_DEPTH_M + 0.01
    )