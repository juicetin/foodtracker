import { requireNativeModule } from 'expo-modules-core';

const FtpClientNative = requireNativeModule('FtpClient');

export const ftpClientModule = {
  upload(
    host: string,
    port: number,
    user: string,
    pass: string,
    remotePath: string,
    localPath: string,
  ): Promise<void> {
    return FtpClientNative.upload(host, port, user, pass, remotePath, localPath);
  },

  download(
    host: string,
    port: number,
    user: string,
    pass: string,
    remotePath: string,
    localPath: string,
  ): Promise<void> {
    return FtpClientNative.download(host, port, user, pass, remotePath, localPath);
  },

  testConnection(
    host: string,
    port: number,
    user: string,
    pass: string,
  ): Promise<boolean> {
    return FtpClientNative.testConnection(host, port, user, pass);
  },
};
