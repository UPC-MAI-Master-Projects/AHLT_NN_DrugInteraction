#!/usr/bin/env bash
#
# Runs Stanford CoreNLP server

# set this path to the directory where you decompressed StanfordCore
STANFORDDIR=stanford-corenlp-4.5.7

if [ -f /tmp/corenlp.shutdown ]; then
    echo "server already running"
else
    # Change to the Stanford NLP directory
    cd "$STANFORDDIR"
    echo "Starting server..."
    echo java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $*
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $* &
    echo $! > /tmp/corenlp-server.running
    wait
    rm /tmp/corenlp-server.running
fi

