package forge_test

// These tests run the client against a gRPC server on a loopback socket: a
// TLS listener that requires and verifies a client certificate, the same
// transport contract as the hosted endpoint, and a plaintext listener. They
// check what the client puts on the wire. The certificates are generated per
// test run.

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	forge "github.com/VoxellInc/forge-go"
	forgev1 "github.com/VoxellInc/forge-go/internal/forgev1"
)

// seen is what the server observed on one call.
type seen struct {
	authorization []string
	clientCN      string // subject CN of the verified client certificate, "" on plaintext
	req           *forgev1.EmbedRequest
}

// recorder is an EmbedService that records each call and, like the hosted
// endpoint, rejects a call that carries no API key.
type recorder struct {
	forgev1.UnimplementedEmbedServiceServer
	mu    sync.Mutex
	calls []seen
}

func (r *recorder) observe(ctx context.Context, req *forgev1.EmbedRequest) (seen, error) {
	s := seen{req: req}
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		s.authorization = md.Get("authorization")
	}
	if p, ok := peer.FromContext(ctx); ok {
		if ti, ok := p.AuthInfo.(credentials.TLSInfo); ok && len(ti.State.VerifiedChains) > 0 {
			s.clientCN = ti.State.VerifiedChains[0][0].Subject.CommonName
		}
	}
	r.mu.Lock()
	r.calls = append(r.calls, s)
	r.mu.Unlock()
	if len(s.authorization) == 0 {
		return s, status.Error(codes.Unauthenticated, "unauthenticated: no api key")
	}
	return s, nil
}

func (r *recorder) Embed(ctx context.Context, req *forgev1.EmbedRequest) (*forgev1.EmbedResponse, error) {
	if _, err := r.observe(ctx, req); err != nil {
		return nil, err
	}
	dim := int(req.Dim)
	if dim == 0 {
		dim = 4
	}
	out := make([]*forgev1.Embedding, len(req.Texts))
	for i := range req.Texts {
		v := make([]float32, dim)
		v[0] = float32(i + 1)
		out[i] = &forgev1.Embedding{Values: v}
	}
	return &forgev1.EmbedResponse{
		Embeddings:  out,
		TotalTokens: int32(3 * len(req.Texts)),
		Model:       "label-from-server",
		Dim:         int32(dim),
		LatencyMs:   7,
		RequestId:   req.RequestId,
	}, nil
}

func (r *recorder) Health(ctx context.Context, _ *forgev1.HealthRequest) (*forgev1.HealthResponse, error) {
	if _, err := r.observe(ctx, nil); err != nil {
		return nil, err
	}
	return &forgev1.HealthResponse{Status: "ok", Models: []string{"turbo"}, UptimeSeconds: 42}, nil
}

func (r *recorder) last(t *testing.T) seen {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		t.Fatal("server saw no call")
	}
	return r.calls[len(r.calls)-1]
}

// pki is a throwaway CA with one server leaf and one client leaf.
type pki struct {
	pool       *x509.CertPool
	serverCert tls.Certificate
	certFile   string // client certificate, PEM
	keyFile    string // client key, PEM
	caFile     string // CA certificate, PEM
}

func newPKI(t *testing.T) *pki {
	t.Helper()
	now := time.Now()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "forge-go test CA"},
		NotBefore:             now.Add(-time.Hour),
		NotAfter:              now.Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	caCert, err := x509.ParseCertificate(caDER)
	if err != nil {
		t.Fatal(err)
	}
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})

	leaf := func(serial int64, cn string, usage x509.ExtKeyUsage, server bool) (certPEM, keyPEM []byte) {
		key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		tmpl := &x509.Certificate{
			SerialNumber: big.NewInt(serial),
			Subject:      pkix.Name{CommonName: cn},
			NotBefore:    now.Add(-time.Hour),
			NotAfter:     now.Add(time.Hour),
			KeyUsage:     x509.KeyUsageDigitalSignature,
			ExtKeyUsage:  []x509.ExtKeyUsage{usage},
		}
		if server {
			tmpl.DNSNames = []string{"localhost"}
			tmpl.IPAddresses = []net.IP{net.ParseIP("127.0.0.1")}
		}
		der, err := x509.CreateCertificate(rand.Reader, tmpl, caCert, &key.PublicKey, caKey)
		if err != nil {
			t.Fatal(err)
		}
		keyDER, err := x509.MarshalPKCS8PrivateKey(key)
		if err != nil {
			t.Fatal(err)
		}
		return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
			pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	}

	srvCertPEM, srvKeyPEM := leaf(2, "localhost", x509.ExtKeyUsageServerAuth, true)
	serverCert, err := tls.X509KeyPair(srvCertPEM, srvKeyPEM)
	if err != nil {
		t.Fatal(err)
	}
	cliCertPEM, cliKeyPEM := leaf(3, "forge-go test client", x509.ExtKeyUsageClientAuth, false)

	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	pool := x509.NewCertPool()
	pool.AddCert(caCert)
	return &pki{
		pool:       pool,
		serverCert: serverCert,
		certFile:   write("client.crt", cliCertPEM),
		keyFile:    write("client.key", cliKeyPEM),
		caFile:     write("ca.crt", caPEM),
	}
}

// serve starts a gRPC server on 127.0.0.1 and returns its port.
func serve(t *testing.T, rec *recorder, opts ...grpc.ServerOption) string {
	t.Helper()
	lis, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer(opts...)
	forgev1.RegisterEmbedServiceServer(srv, rec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	_, port, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	return port
}

// serveMTLS starts a server that requires and verifies a client certificate.
func serveMTLS(t *testing.T, rec *recorder, p *pki) string {
	t.Helper()
	return serve(t, rec, grpc.Creds(credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{p.serverCert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    p.pool,
		MinVersion:   tls.VersionTLS12,
	})))
}

func ctx5(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// Over mutual TLS the client presents its certificate and still sends the API
// key, on Embed and on Health. The hosted endpoint needs both.
func TestMTLSSendsCertificateAndAPIKey(t *testing.T) {
	p := newPKI(t)
	rec := &recorder{}
	port := serveMTLS(t, rec, p)

	c, err := forge.NewClient(forge.Options{
		Address:  "127.0.0.1:" + port,
		CertFile: p.certFile,
		KeyFile:  p.keyFile,
		CAFile:   p.caFile,
		APIKey:   "test-key",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()
	if c.Auth() != forge.AuthMTLS {
		t.Fatalf("Auth() = %q, want %q", c.Auth(), forge.AuthMTLS)
	}

	res, err := c.Embed(ctx5(t), []string{"a", "b"},
		forge.WithModel("pro"), forge.WithDim(8), forge.WithInputType("query"))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(res.Embeddings) != 2 || len(res.Embeddings[0]) != 8 || res.Dim != 8 {
		t.Fatalf("got %d vectors of len %d (Dim=%d), want 2 of len 8", len(res.Embeddings), len(res.Embeddings[0]), res.Dim)
	}
	if res.Embeddings[1][0] != 2 || res.Tokens != 6 || res.Model != "pro" || res.LatencyMs != 7 {
		t.Fatalf("response not mapped through: %+v", res)
	}

	got := rec.last(t)
	if got.clientCN != "forge-go test client" {
		t.Fatalf("server verified client CN %q, want the test client certificate", got.clientCN)
	}
	if len(got.authorization) != 1 || got.authorization[0] != "Bearer test-key" {
		t.Fatalf("authorization metadata = %q, want [\"Bearer test-key\"]", got.authorization)
	}
	if got.req.Model != "pro" || got.req.Dim != 8 || got.req.InputType != "query" || got.req.RequestId == "" {
		t.Fatalf("request not built from options: %+v", got.req)
	}

	st, models, up, err := c.Health(ctx5(t))
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if st != "ok" || len(models) != 1 || up != 42 {
		t.Fatalf("Health = %q %v %d", st, models, up)
	}
	if h := rec.last(t); len(h.authorization) != 1 || h.authorization[0] != "Bearer test-key" {
		t.Fatalf("Health authorization metadata = %q, want the API key", h.authorization)
	}
}

// With a certificate but no API key the client sends no authorization
// metadata, and an endpoint that authorizes by key rejects the call.
func TestMTLSWithoutAPIKeyIsRejected(t *testing.T) {
	p := newPKI(t)
	rec := &recorder{}
	port := serveMTLS(t, rec, p)

	c, err := forge.NewClient(forge.Options{
		Address:  "127.0.0.1:" + port,
		CertFile: p.certFile,
		KeyFile:  p.keyFile,
		CAFile:   p.caFile,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	_, err = c.Embed(ctx5(t), []string{"a"})
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("Embed error = %v, want Unauthenticated", err)
	}
	if got := rec.last(t); len(got.authorization) != 0 || got.clientCN == "" {
		t.Fatalf("server saw authorization=%q clientCN=%q, want no key and a verified certificate", got.authorization, got.clientCN)
	}
}

// A hostname target goes through the IPv6-first dialer. The server listens on
// IPv4 only, as the hosted endpoint does, so the IPv4 fallback has to work.
func TestHostnameTargetFallsBackToIPv4(t *testing.T) {
	p := newPKI(t)
	rec := &recorder{}
	port := serveMTLS(t, rec, p)

	c, err := forge.NewClient(forge.Options{
		Address:  "localhost:" + port,
		CertFile: p.certFile,
		KeyFile:  p.keyFile,
		CAFile:   p.caFile,
		APIKey:   "test-key",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	if _, err := c.Embed(ctx5(t), []string{"a"}); err != nil {
		t.Fatalf("Embed via hostname: %v", err)
	}
	if got := rec.last(t); got.req.Model != "turbo" {
		t.Fatalf("default model = %q, want turbo", got.req.Model)
	}
}

// A server that requires a client certificate refuses a client whose
// certificate was issued by a different CA.
func TestMTLSRejectsCertificateFromAnotherCA(t *testing.T) {
	p := newPKI(t)
	other := newPKI(t)
	rec := &recorder{}
	port := serveMTLS(t, rec, p)

	c, err := forge.NewClient(forge.Options{
		Address:  "127.0.0.1:" + port,
		CertFile: other.certFile, // not signed by the server's CA
		KeyFile:  other.keyFile,
		CAFile:   p.caFile,
		APIKey:   "test-key",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	if _, err := c.Embed(ctx5(t), []string{"a"}); err == nil {
		t.Fatal("Embed succeeded with a certificate the server does not trust")
	}
	rec.mu.Lock()
	n := len(rec.calls)
	rec.mu.Unlock()
	if n != 0 {
		t.Fatalf("server handled %d call(s) from an untrusted certificate, want 0", n)
	}
}

// Plaintext mode for a local endpoint: API key only.
func TestInsecureSendsAPIKey(t *testing.T) {
	rec := &recorder{}
	port := serve(t, rec)

	c, err := forge.NewClient(forge.Options{Address: "127.0.0.1:" + port, APIKey: "local-key", Insecure: true})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()
	if c.Auth() != forge.AuthAPIKey {
		t.Fatalf("Auth() = %q, want %q", c.Auth(), forge.AuthAPIKey)
	}

	if _, err := c.Embed(ctx5(t), []string{"a"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	got := rec.last(t)
	if len(got.authorization) != 1 || got.authorization[0] != "Bearer local-key" {
		t.Fatalf("authorization metadata = %q, want [\"Bearer local-key\"]", got.authorization)
	}
	if got.clientCN != "" {
		t.Fatalf("plaintext call reported a client certificate %q", got.clientCN)
	}
}
