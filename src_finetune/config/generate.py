import os


if __name__ == "__main__":
    lines = []
    dir = "/work/pi_mchiovaro_uri_edu/richard_buckley/musicgen/jsonl_data"
    with open("7B.yaml", "r") as f:
        # find the instruct_data key
        i = 0
        csf = None
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "instruct_data: " not in line:
                continue

            # split the line into key and value
            key, value = line.split(": ")

            # find all files within the directory
            files = [f for f in os.listdir(dir) if f.endswith(".jsonl")]
            csf = ",".join(files)
            break

        if csf is None:
            raise ValueError("instruct_data not found")

        # replace the line with the new value
        lines[i] = lines[i].split(": ")[0] + ": " + f'"{dir}/{csf}"' + "\n"

    with open("7B.yaml", "w") as f:
        f.writelines(lines)

    print("✅ Done!")
