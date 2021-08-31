# Basic config
DEPS=./config ./Dockerfile.base
SLAVE_DEPS=$(DEPS) ./Dockerfile.leaf
DEFAULT_RUN_FLAGS=-id --rm --runtime=nvidia
NOBLE_NETWORK=host
SLAVE_NETWORK=host
DOCKER_BASE=./Dockerfile.base
DOCKER_LEAF=./Dockerfile.leaf

# Creates the tagname of the images from the branch of the git
BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD)
ifeq ($(BRANCH_NAME), master)
	TAG_NAME=latest
else
	TAG_NAME=$(BRANCH_NAME)
endif


### Build section ###


BASE=mime-base
MAIN=mime-brain mime-capture mime-face mime-limbs
DEV=mime-terminal

DEFAULT_OPTIONS=--target="$@" --tag="$@:$(TAG_NAME)" --build-arg TAGN=$(TAG_NAME)
BASE_OPTIONS=$(DEFAULT_OPTIONS) --network="$(NOBLE_NETWORK)"
BRAIN_OPTIONS=$(BASE_OPTIONS)
__DEFAULT_LEAF_OPTIONS=$(DEFAULT_OPTIONS)
NORMAL_LEAF_OPTIONS=$(__DEFAULT_LEAF_OPTIONS) --network="$(SLAVE_NETWORK)"
DEV_LEAF_OPTIONS=$(__DEFAULT_LEAF_OPTIONS) --network="$(NOBLE_NETWORK)" --no-cache


all: $(BASE) $(MAIN) $(DEV)
main: $(BASE) $(MAIN)
dev: $(BASE) $(DEV)


mime-base: $(DEPS)
	docker build $(BASE_OPTIONS) -f $(DOCKER_BASE) .

mime-brain: $(SLAVE_DEPS) | mime-base
	docker build $(BRAIN_OPTIONS) -f $(DOCKER_LEAF) .

mime-capture: ./perception $(SLAVE_DEPS) | mime-base
	docker build $(NORMAL_LEAF_OPTIONS) -f $(DOCKER_LEAF) .

mime-face: ./face $(SLAVE_DEPS) | mime-base
	docker build $(NORMAL_LEAF_OPTIONS) -f $(DOCKER_LEAF) .

mime-limbs: ./limbs $(SLAVE_DEPS) | mime-base
	docker build $(NORMAL_LEAF_OPTIONS) -f $(DOCKER_LEAF) .

mime-terminal: ./terminal $(SLAVE_DEPS) | mime-base
	docker build $(DEV_LEAF_OPTIONS) -f $(DOCKER_LEAF) .


### RUN SECTION ###


# TODO: This


### CLEAN SECTION ###


.PHONY: clean
clean:
	docker container prune
	for dImage in $(BASE) $(MAIN) $(DEV) ; do \
		docker rmi -f $$dImage:$(TAG_NAME) ; \
	done
	rm -rf perception/vpiinterop/build/*
	docker image prune
	
