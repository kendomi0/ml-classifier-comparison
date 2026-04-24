def get_user_choice(dictionary):
    dict_str = ", ".join(dictionary)
    current_dataset = None
    while current_dataset is None:
        user_input = input(f"Type in which dataset you'd like to use ({dict_str}): ").lower()
        if user_input not in dictionary:
            print("Invalid input.")
        else:
            current_dataset = user_input
    return current_dataset

def do_nothing(var):
    return var