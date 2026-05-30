# forge-go

Go client for [**Forge**](https://voxell.ai/forge) — Voxell's hosted text-embedding API.

Native **gRPC + protobuf** transport with **mutual-TLS** auth — not HTTP/JSON. Embedding responses
are dense float arrays; protobuf packs them as binary (4 bytes/float) instead of JSON text, so the
vector payload is smaller and parses faster. Auth is cert-based (client certificates), not a
long-lived bearer token on the wire.

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

	forge "github.com/VoxellInc/forge-go"
)

func main() {
	c, err := forge.NewClient(forge.Options{
		Address:  forge.DefaultAddress, // forge-control.fly.dev:50052
		CertFile: "~/.forge/client.crt",
		KeyFile:  "~/.forge/client.key",
		CAFile:   "~/.forge/voxell-ca.crt",
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
| `ultra` | 4096 | Qwen3-Embedding-8B; ~75+ avg task score on MTEB, currently #4 on MTEB (English) |

### Matryoshka (shorter vectors)

```go
res, _ := c.Embed(ctx, texts, forge.WithModel("turbo"), forge.WithDim(256)) // re-normalized 256-d
```

## Authentication

- **mTLS (hosted endpoint):** provision a client certificate with the Forge CLI
  (`forge-cli auth init --api-key <KEY>` writes `~/.forge/{client.crt,client.key,voxell-ca.crt}`),
  then point `Options` at those files.
- **API key (local/plaintext gRPC bridge):** set `APIKey` + `Insecure: true`.

## License

MIT © Voxell, Inc.
