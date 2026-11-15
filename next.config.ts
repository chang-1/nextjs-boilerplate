import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/jem-data/:path*", destination: "https://assets.jorgechang.xyz/seed1-batch4/data/:path*" }];
  },
};

export default nextConfig;
