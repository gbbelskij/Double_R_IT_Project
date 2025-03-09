export interface MainProps {
  /**
   * The content to be rendered inside the `Main` component.
   */
  children: React.ReactNode;

  /**
   * Whether to disable the header offset. Optional. Default is `false`.
   */
  disableHeaderOffset?: boolean;
}
