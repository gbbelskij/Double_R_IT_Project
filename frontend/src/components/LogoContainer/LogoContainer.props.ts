export interface LogoContainerProps {
  /**
   * The content to be rendered inside the LogoContainer.
   */
  children: React.ReactNode;
  /**
   * The offset of the logo in the **number** of pixels.
   * If the value is `false`, it means it is centered.
   */
  logoOffset?: number;
}
