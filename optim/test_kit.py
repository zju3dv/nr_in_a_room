def get_test_kit(name):
    if name == "Beechwood_i_int_bedroom_0":
        assert False
        # active_instance_id = [46, 4]
        # active_instance_id = [46, 4, 102, 48, 9, 10]
        # three furniture
        # active_instance_id = [46, 102, 9]
        # with 2 lamps
        # active_instance_id = [46, 3, 8, 102]
        # active_instance_id = [4]
        # active_instance_id = [46]
        # active_instance_id = [46, 4, 102]
        active_instance_id = [46, 4, 9, 102]
        # door: 102
        # active_instance_id = [102]
        # active_instance_id = [46, 4, 9]
        # active_instance_id = [46]
        relation_info = {
            "46": {
                "attach_wall": True,
                "attach_floor": True,
            },
            "4": {
                "attach_wall": True,
                "attach_floor": True,
            },
            "9": {
                "attach_wall": True,
                "attach_floor": True,
            },
            "102": {
                "attach_wall": True,
                "object_type": "door",
                "attach_floor": True,
            },
        }

    elif name == "Rs_int_living_room_0":
        active_instance_id = [12, 13, 14, 15, 5]
        # active_instance_id = [12]
        # active_instance_id = [5]
        relation_info = {
            "5": {
                "attach_wall": True,
                "attach_floor": True,
            },
            "12": {
                "attach_floor": True,
            },
            "13": {
                "attach_floor": True,
            },
            "14": {
                "attach_floor": True,
            },
            "15": {
                "attach_floor": True,
            },
        }
    elif name == "Beechwood_1_int_bedroom_0_no_random":
        # active_instance_id = [48, 4, 9, 104]
        active_instance_id = [48, 4, 9, 104, 97, 94, 62, 64, 51, 5, 3, 10, 50, 49]
        # active_instance_id = [48, 4, 9, 104, 94, 62, 64, 51, 5, 3, 10, 50, 49]
        # door: 102
        relation_info = {}
    elif name == "Beechwood_0_int_lobby_0":
        # active_instance_id = [4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 96, 107, 108, 110, 111]  # fmt: skip
        # 4 piano, 31 large desk
        active_instance_id = [4, 31]
        relation_info = {}

    return active_instance_id, relation_info
