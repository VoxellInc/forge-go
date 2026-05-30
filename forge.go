// Package forge is a Go client for Forge — Voxell's hosted text-embedding API.
//
// It speaks the native gRPC + protobuf transport (not HTTP/JSON): embedding
// responses are dense float arrays, which protobuf packs as binary rather than
// JSON text, so payloads are smaller and parse faster. Auth is mutual TLS
// (client certificates) for the hosted endpoint, with an API-key fallback for
// local/plaintext gRPC.
//
// Example:
//
//	c, err := forge.NewClient(forge.Options{
//		Address:  "forge-control.fly.dev:50052",
//		CertFile: "~/.forge/client.crt",
//		KeyFile:  "~/.forge/client.key",
//		CAFile:   "~/.forge/voxell-ca.crt",
//	})
//	if err != nil { log.Fatal(err) }
//	defer c.Close()
//
//	res, err := c.Embed(ctx, []string{"hello world"}, forge.WithModel("turbo"))
package forge

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"

	forgev1 "github.com/VoxellInc/forge-go/internal/forgev1"
)

// AuthMethod reports how a Client authenticated.
type AuthMethod string

const (
	AuthMTLS   AuthMethod = "mtls"
	AuthAPIKey AuthMethod = "api_key"
)

// DefaultAddress is the hosted Forge control-plane gRPC endpoint.
const DefaultAddress = "forge-control.fly.dev:50052"

// Options configures a Client. Provide either mTLS cert files (hosted endpoint)
// or an APIKey with Insecure=true (local/plaintext gRPC bridge).
type Options struct {
	Address string // host:port; defaults to DefaultAddress

	// mTLS (preferred, for the hosted endpoint):
	CertFile string
	KeyFile  string
	CAFile   string

	// API-key fallback (for a local/plaintext gRPC endpoint):
	APIKey   string
	Insecure bool

	// DialTimeout bounds the initial connection (default 10s).
	DialTimeout time.Duration
}

// Client is a Forge gRPC client. Safe for concurrent use.
type Client struct {
	conn   *grpc.ClientConn
	grpc   forgev1.EmbedServiceClient
	apiKey string
	auth   AuthMethod
}

// EmbedResult is the result of an Embed call.
type EmbedResult struct {
	Embeddings [][]float32
	Tokens     int32
	Model      string
	Dim        int32
	LatencyMs  int64
}

// Auth reports the authentication method negotiated at NewClient.
func (c *Client) Auth() AuthMethod { return c.auth }

// NewClient dials Forge with mTLS (if cert files are set) or an API key.
//
// For hostname targets it prefers IPv6 (falling back to IPv4) to avoid shared
// IPv4 cross-region routing.
func NewClient(opts Options) (*Client, error) {
	addr := opts.Address
	if addr == "" {
		addr = DefaultAddress
	}
	timeout := opts.DialTimeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}

	dialOpts := []grpc.DialOption{
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64 << 20)),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                20 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	}

	// Prefer IPv6 for hostname targets (raw IPs are left as-is).
	if host, _, _ := net.SplitHostPort(addr); host != "" && net.ParseIP(host) == nil {
		dialOpts = append(dialOpts, grpc.WithContextDialer(func(ctx context.Context, a string) (net.Conn, error) {
			d := &net.Dialer{Timeout: timeout}
			conn, err := d.DialContext(ctx, "tcp6", a)
			if err != nil {
				conn, err = d.DialContext(ctx, "tcp4", a)
			}
			return conn, err
		}))
	}

	var auth AuthMethod
	switch {
	case opts.CertFile != "" && opts.KeyFile != "" && opts.CAFile != "":
		cert, err := tls.LoadX509KeyPair(expandHome(opts.CertFile), expandHome(opts.KeyFile))
		if err != nil {
			return nil, fmt.Errorf("load client cert: %w", err)
		}
		caPEM, err := os.ReadFile(expandHome(opts.CAFile))
		if err != nil {
			return nil, fmt.Errorf("read CA cert: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("invalid CA certificate in %s", opts.CAFile)
		}
		creds := credentials.NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}, RootCAs: pool})
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(creds))
		auth = AuthMTLS
	case opts.APIKey != "" && opts.Insecure:
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
		auth = AuthAPIKey
	default:
		return nil, fmt.Errorf("forge: no credentials — set CertFile/KeyFile/CAFile (mTLS) or APIKey with Insecure=true")
	}

	conn, err := grpc.NewClient(addr, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", addr, err)
	}
	return &Client{conn: conn, grpc: forgev1.NewEmbedServiceClient(conn), apiKey: opts.APIKey, auth: auth}, nil
}

// EmbedOption customizes an Embed request.
type EmbedOption func(*forgev1.EmbedRequest)

// WithModel sets the tier: "turbo" (1024d), "pro" (2560d), or "ultra" (4096d).
func WithModel(model string) EmbedOption {
	return func(r *forgev1.EmbedRequest) { r.Model = model }
}

// WithDim sets a Matryoshka truncation dimension (re-normalized); 0 = native.
func WithDim(dim int) EmbedOption {
	return func(r *forgev1.EmbedRequest) { r.Dim = int32(dim) }
}

// WithInputType sets "query" or "document".
func WithInputType(t string) EmbedOption {
	return func(r *forgev1.EmbedRequest) { r.InputType = t }
}

// Embed returns vector embeddings for texts. Default model is "turbo".
func (c *Client) Embed(ctx context.Context, texts []string, opts ...EmbedOption) (*EmbedResult, error) {
	req := &forgev1.EmbedRequest{
		Texts:     texts,
		Model:     "turbo",
		RequestId: fmt.Sprintf("%x", time.Now().UnixNano()),
	}
	for _, o := range opts {
		o(req)
	}
	if c.auth == AuthAPIKey && c.apiKey != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.grpc.Embed(ctx, req)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(resp.Embeddings))
	for i, e := range resp.Embeddings {
		out[i] = e.Values
	}
	return &EmbedResult{
		Embeddings: out,
		Tokens:     resp.TotalTokens,
		Model:      resp.Model,
		Dim:        resp.Dim,
		LatencyMs:  resp.LatencyMs,
	}, nil
}

// Health reports engine liveness, loaded models, and uptime.
func (c *Client) Health(ctx context.Context) (status string, models []string, uptimeSeconds int64, err error) {
	resp, err := c.grpc.Health(ctx, &forgev1.HealthRequest{})
	if err != nil {
		return "", nil, 0, err
	}
	return resp.Status, resp.Models, resp.UptimeSeconds, nil
}

// Close releases the underlying connection.
func (c *Client) Close() error { return c.conn.Close() }

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}
