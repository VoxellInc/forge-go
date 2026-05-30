package forge_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	forge "github.com/VoxellInc/forge-go"
)

func TestNoCredentialsErrors(t *testing.T) {
	if _, err := forge.NewClient(forge.Options{Address: "x:1"}); err == nil {
		t.Fatal("expected error with no credentials")
	}
}

func homeForge(f string) string {
	h, _ := os.UserHomeDir()
	return filepath.Join(h, ".forge", f)
}

// liveClient dials the hosted endpoint with the local ~/.forge mTLS certs.
// Skips when certs are absent (CI / other machines).
func liveClient(t *testing.T) *forge.Client {
	t.Helper()
	cert := homeForge("client.crt")
	if _, err := os.Stat(cert); err != nil {
		t.Skip("no ~/.forge/client.crt — skipping live gRPC test")
	}
	c, err := forge.NewClient(forge.Options{
		Address:  forge.DefaultAddress,
		CertFile: cert,
		KeyFile:  homeForge("client.key"),
		CAFile:   homeForge("voxell-ca.crt"),
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c
}

func TestLiveEmbedTiers(t *testing.T) {
	c := liveClient(t)
	defer c.Close()
	for tier, dim := range map[string]int{"turbo": 1024, "pro": 2560, "ultra": 4096} {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		res, err := c.Embed(ctx, []string{"hello", "world"}, forge.WithModel(tier), forge.WithInputType("document"))
		cancel()
		if err != nil {
			t.Fatalf("%s embed: %v", tier, err)
		}
		if len(res.Embeddings) != 2 {
			t.Fatalf("%s: want 2 vectors, got %d", tier, len(res.Embeddings))
		}
		if got := len(res.Embeddings[0]); got != dim {
			t.Fatalf("%s: want dim %d, got %d", tier, dim, got)
		}
		t.Logf("%s: 2 vecs dim=%d tokens=%d engine_ms=%d auth=%s", tier, res.Dim, res.Tokens, res.LatencyMs, c.Auth())
	}
}

func TestLiveMatryoshka(t *testing.T) {
	c := liveClient(t)
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	res, err := c.Embed(ctx, []string{"truncate me"}, forge.WithModel("turbo"), forge.WithDim(256))
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if got := len(res.Embeddings[0]); got != 256 {
		t.Fatalf("want dim 256, got %d", got)
	}
}

func TestLiveHealth(t *testing.T) {
	c := liveClient(t)
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	status, models, up, err := c.Health(ctx)
	if err != nil {
		t.Logf("Health not available on this endpoint (ok): %v", err)
		return
	}
	t.Logf("health: %s models=%v uptime=%ds", status, models, up)
}
