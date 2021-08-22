# Basic config
DEPS=Dockerfile ./config
DEFAULT_RUN_FLAGS=-id --rm --runtime=nvidia
NOBLE_NETWORK=host
SLAVE_NETWORK=host

# Creates the tagname of the images from the branch of the git
BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD)
ifeq ($(BRANCH_NAME), master)
	TAG_NAME=latest
else
	TAG_NAME=$(BRANCH_NAME)
endif


### Build section ###

ALL=mime-base mime-brain mime-terminal mime-capture mime-face mime-audio mime-limbs
all: $(ALL)

mime-base: $(DEPS)
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(NOBLE_NETWORK)" .

mime-brain: $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(NOBLE_NETWORK)" .

mime-terminal: ./terminal $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(NOBLE_NETWORK)" .

mime-capture: ./perception $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(SLAVE_NETWORK)" .

mime-face: ./face $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(SLAVE_NETWORK)" .

mime-audio: ./audio $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(SLAVE_NETWORK)" .

mime-limbs: ./limbs $(DEPS) | mime-base
	docker build --target="$@" --tag="$@:$(TAG_NAME)" --network="$(SLAVE_NETWORK)" .


### RUN SECTION ###


# TODO: This


### CLEAN SECTION ###


.PHONY: clean
clean:
	docker container prune
	docker image prune
	docker rmi $(ALL)
