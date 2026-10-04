# forge-go

Go client for [**Forge**](https://voxell.ai/forge), Voxell's hosted text-embedding API.

Voxell's Ingot-8B-R3 ranks #1 for English on the public MTEB leaderboard (English v2), with a 75.98
mean task score across 41 tasks. It is the top usable English embedding model. See the
[model card](https://huggingface.co/JCorners/Ingot-8B-R3), or try Forge with no signup on the
[playground](https://playground.voxell.ai).

Voxell also publishes retrieval receipts measured on four public corpora: SEC filings (2,010
documents), USPTO patents (4,008), NASA technical reports (1,210) and arXiv technical papers (589).
That is 7,817 documents and 980,885 passages, with 200 questions per corpus, measured 2026-10-02.
Across the 800 questions the first result answers the question for 85%, one of the top three
results answers it for 91%, and the right document is in the top ten for 95%. The questions were
written by a model from the documents and judged against the passage text, which is easier than a
human test set. The figures describe what the full retrieval pipeline does on these corpora, not
the embedding call this client makes, and they are not a comparison with any other vendor.
[Read the receipts](https://voxell.ai/retrieval/).

Native **gRPC + protobuf** transport over **mutual TLS** — not HTTP/JSON. Embedding responses
are dense float arrays; protobuf packs them as binary (4 bytes/float) instead of JSON text, so the
vector payload is smaller and parses faster. A client certificate identifies the connection and
your API key authorizes each request.

## Install

```bash
go get github.com/VoxellInc/forge-go
```

## Usage

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	forge "github.com/VoxellInc/forge-go"
)

func main() {
	c, err := forge.NewClient(forge.Options{
		Address:  forge.DefaultAddress, // edge.voxell.ai:8443
		CertFile: "~/.forge/client.crt",
		KeyFile:  "~/.forge/client.key",
		CAFile:   "~/.forge/voxell-ca.crt",
		APIKey:   os.Getenv("FORGE_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	res, err := c.Embed(context.Background(),
		[]string{"the quick brown fox", "lazy dog"},
		forge.WithModel("turbo"),          // turbo | pro | ultra
		forge.WithInputType("document"),   // or "query"
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d vectors, dim=%d, tokens=%d\n", len(res.Embeddings), res.Dim, res.Tokens)
}
```

### Models

| Model | Dim | Notes |
| ----- | --- | ----- |
| `turbo` | 1024 | fast, low cost (default) |
| `pro` | 2560 | |
| `ultra` | 4096 | highest quality |

### Matryoshka (shorter vectors)

```go
res, _ := c.Embed(ctx, texts, forge.WithModel("turbo"), forge.WithDim(256)) // re-normalized 256-d
```

## Authentication

The hosted endpoint (`edge.voxell.ai:8443`) takes a client certificate and an API key together.

- **Client certificate (mutual TLS):** provision one with the Forge CLI
  (`forge-cli auth init --api-key <KEY>` writes `~/.forge/{client.crt,client.key,voxell-ca.crt}`),
  then point `Options` at those files. It identifies the connection. Certificates last 90 days;
  re-run `auth init` to renew.
- **API key:** set `APIKey`. It is sent as a bearer token in request metadata, inside the TLS
  channel, and authorizes each request.

For a local plaintext gRPC endpoint, set `APIKey` + `Insecure: true` and no certificate.

## License

MIT © Voxell, Inc.
