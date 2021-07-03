import fileinput
import binascii

for line in fileinput.input():
    line = line[:-1].encode('utf8')
    crc = 0xFFF & binascii.crc32(line)
    print(f'{line}: [{hex(crc)}]')
